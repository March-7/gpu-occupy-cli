use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use dialoguer::{Confirm, Input, MultiSelect, Select, theme::ColorfulTheme};
use std::cmp::Reverse;
use std::env;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fs;
#[cfg(unix)]
use std::os::unix::fs as unix_fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_RESERVE: &str = "1GiB";
const DEFAULT_CHUNK_SIZE: &str = "256MiB";
const DEFAULT_STATUS_INTERVAL: u64 = 5;

#[derive(Parser, Debug)]
#[command(
    name = "gpu-occupy",
    version,
    about = "Keep NVIDIA GPU memory occupied from a Rust CLI.",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Show current GPU memory usage and utilization.
    List,
    /// Hold GPU memory until Ctrl+C.
    Hold(HoldArgs),
    /// Install this binary into ~/.local/bin/gpu-occupy by default.
    Install(InstallArgs),
}

#[derive(Args, Debug, Clone)]
struct HoldArgs {
    /// GPU index. If omitted, the tool picks the emptiest GPU that can satisfy the request.
    #[arg(long)]
    gpu: Option<usize>,

    /// Amount of memory to allocate, such as 20GiB, 500MiB, 20000MB.
    #[arg(long, short = 'm')]
    memory: String,

    /// Memory to leave unallocated on the target GPU.
    #[arg(long, default_value = DEFAULT_RESERVE)]
    reserve: String,

    /// Chunk size per allocation request.
    #[arg(long, default_value = DEFAULT_CHUNK_SIZE)]
    chunk_size: String,

    /// Seconds between status prints while holding memory.
    #[arg(long, default_value_t = DEFAULT_STATUS_INTERVAL)]
    status_interval: u64,
}

#[derive(Args, Debug)]
struct InstallArgs {
    /// Target directory for the installed command.
    #[arg(long)]
    bin_dir: Option<PathBuf>,

    /// Replace an existing file or symlink at the destination.
    #[arg(long)]
    force: bool,
}

#[derive(Clone, Debug)]
struct GpuInfo {
    index: usize,
    name: String,
    total_mib: u64,
    used_mib: u64,
    util_pct: u32,
}

impl GpuInfo {
    fn free_mib(&self) -> u64 {
        self.total_mib.saturating_sub(self.used_mib)
    }
}

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuDevicePtr = u64;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CuResult;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int) -> CuResult;
    fn cuCtxCreate_v2(ctx: *mut CuContext, flags: c_uint, dev: CuDevice) -> CuResult;
    fn cuCtxDestroy_v2(ctx: CuContext) -> CuResult;
    fn cuCtxSetCurrent(ctx: CuContext) -> CuResult;
    fn cuMemAlloc_v2(dptr: *mut CuDevicePtr, bytesize: usize) -> CuResult;
    fn cuMemFree_v2(dptr: CuDevicePtr) -> CuResult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CuResult;
    fn cuMemsetD8_v2(dst: CuDevicePtr, uc: u8, n: usize) -> CuResult;
    fn cuGetErrorString(error: CuResult, p_str: *mut *const c_char) -> CuResult;
}

struct CudaAllocation {
    ctx: CuContext,
    ptrs: Vec<CuDevicePtr>,
    allocated_bytes: u64,
}

struct ActiveHold {
    gpu: GpuInfo,
    allocation: CudaAllocation,
    target_bytes: u64,
}

impl CudaAllocation {
    fn new(gpu_index: usize) -> Result<Self> {
        unsafe { cuda_check(cuInit(0), "cuInit")? };

        let mut device = 0;
        unsafe { cuda_check(cuDeviceGet(&mut device, gpu_index as c_int), "cuDeviceGet")? };

        let mut ctx = std::ptr::null_mut();
        unsafe {
            cuda_check(cuCtxCreate_v2(&mut ctx, 0, device), "cuCtxCreate_v2")?;
        }

        Ok(Self {
            ctx,
            ptrs: Vec::new(),
            allocated_bytes: 0,
        })
    }

    fn memory_info(&self) -> Result<(u64, u64)> {
        self.make_current()?;
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe {
            cuda_check(cuMemGetInfo_v2(&mut free, &mut total), "cuMemGetInfo_v2")?;
        }
        Ok((free as u64, total as u64))
    }

    fn allocate(&mut self, bytes: u64) -> Result<()> {
        self.make_current()?;
        let bytes_usize =
            usize::try_from(bytes).context("requested size exceeds platform limits")?;
        let mut ptr = 0u64;
        unsafe {
            cuda_check(cuMemAlloc_v2(&mut ptr, bytes_usize), "cuMemAlloc_v2")?;
            let touch = bytes_usize.min(MIB as usize);
            cuda_check(cuMemsetD8_v2(ptr, 0, touch), "cuMemsetD8_v2")?;
        }
        self.ptrs.push(ptr);
        self.allocated_bytes += bytes;
        Ok(())
    }

    fn make_current(&self) -> Result<()> {
        unsafe { cuda_check(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent") }
    }
}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        let _ = self.make_current();
        for ptr in self.ptrs.drain(..).rev() {
            unsafe {
                let _ = cuMemFree_v2(ptr);
            }
        }

        if !self.ctx.is_null() {
            unsafe {
                let _ = cuCtxSetCurrent(std::ptr::null_mut());
                let _ = cuCtxDestroy_v2(self.ctx);
            }
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::List) => list_gpus(),
        Some(Commands::Hold(args)) => hold_memory(args),
        Some(Commands::Install(args)) => install_binary(args),
        None => interactive_menu(),
    }
}

fn interactive_menu() -> Result<()> {
    let theme = ColorfulTheme::default();
    let actions = [
        "Hold GPU memory",
        "List GPUs",
        "Install gpu-occupy into ~/.local/bin",
        "Exit",
    ];

    let choice = Select::with_theme(&theme)
        .with_prompt("gpu-occupy")
        .default(0)
        .items(actions)
        .interact()
        .context("interactive terminal is required")?;

    match choice {
        0 => run_interactive_hold(&theme),
        1 => list_gpus(),
        2 => install_binary(InstallArgs {
            bin_dir: None,
            force: true,
        }),
        _ => Ok(()),
    }
}

fn run_interactive_hold(theme: &ColorfulTheme) -> Result<()> {
    let gpus = query_gpus()?;
    print_gpu_table(&gpus);

    let gpu_options = gpus.iter().map(format_gpu_option).collect::<Vec<_>>();
    let gpu_choices = MultiSelect::with_theme(theme)
        .with_prompt("Choose GPU(s). Leave empty to auto-select one GPU")
        .items(&gpu_options)
        .interact()?;

    let memory_gib = Input::<f64>::with_theme(theme)
        .with_prompt("How many GiB to hold")
        .default(20.0)
        .interact_text()?;

    let args = HoldArgs {
        gpu: None,
        memory: format_float_unit(memory_gib, "GiB"),
        reserve: DEFAULT_RESERVE.to_owned(),
        chunk_size: DEFAULT_CHUNK_SIZE.to_owned(),
        status_interval: DEFAULT_STATUS_INTERVAL,
    };
    let selected_gpus = gpu_choices
        .into_iter()
        .map(|choice| gpus[choice].index)
        .collect::<Vec<_>>();

    println!();
    println!("Plan");
    println!(
        "GPU: {}",
        if selected_gpus.is_empty() {
            "auto-select one GPU".to_owned()
        } else {
            selected_gpus
                .iter()
                .map(|idx| idx.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!("Memory: {}", args.memory);

    let confirmed = Confirm::with_theme(theme)
        .with_prompt("Start occupying GPU memory")
        .default(true)
        .interact()?;

    if confirmed {
        hold_memory_on_gpus(args, selected_gpus)?;
    }

    Ok(())
}

fn format_gpu_option(gpu: &GpuInfo) -> String {
    format!(
        "GPU {} | {} | used {} | free {} | util {}%",
        gpu.index,
        gpu.name,
        format_mib(gpu.used_mib),
        format_mib(gpu.free_mib()),
        gpu.util_pct
    )
}

fn format_float_unit(value: f64, unit: &str) -> String {
    let rounded = (value * 100.0).round() / 100.0;
    if (rounded - rounded.trunc()).abs() < f64::EPSILON {
        format!("{:.0}{unit}", rounded)
    } else {
        format!("{rounded:.2}{unit}")
    }
}

fn install_binary(args: InstallArgs) -> Result<()> {
    let current_exe = env::current_exe().context("failed to locate current executable")?;
    let current_exe = fs::canonicalize(current_exe).context("failed to canonicalize executable")?;

    let bin_dir = match args.bin_dir {
        Some(path) => path,
        None => home_dir()?.join(".local/bin"),
    };
    fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed to create {}", bin_dir.display()))?;

    let destination = bin_dir.join("gpu-occupy");

    if destination.exists() || fs::symlink_metadata(&destination).is_ok() {
        let same_target = fs::canonicalize(&destination)
            .map(|existing| existing == current_exe)
            .unwrap_or(false);
        if same_target {
            println!(
                "gpu-occupy is already installed at {}",
                destination.display()
            );
            return Ok(());
        }
        if !args.force {
            bail!(
                "{} already exists. Re-run with --force to replace it.",
                destination.display()
            );
        }
        fs::remove_file(&destination)
            .with_context(|| format!("failed to remove {}", destination.display()))?;
    }

    #[cfg(unix)]
    unix_fs::symlink(&current_exe, &destination).with_context(|| {
        format!(
            "failed to create symlink {} -> {}",
            destination.display(),
            current_exe.display()
        )
    })?;

    #[cfg(not(unix))]
    bail!("install is currently implemented for Unix-like systems only");

    println!(
        "Installed gpu-occupy -> {}",
        destination.as_path().display()
    );

    let path_entries = env::var_os("PATH")
        .map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .unwrap_or_default();
    if !path_entries.iter().any(|entry| entry == &bin_dir) {
        println!(
            "Warning: {} is not currently in PATH",
            bin_dir.as_path().display()
        );
    }

    Ok(())
}

fn home_dir() -> Result<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("HOME is not set"))
}

fn list_gpus() -> Result<()> {
    let gpus = query_gpus()?;
    print_gpu_table(&gpus);
    Ok(())
}

fn hold_memory(args: HoldArgs) -> Result<()> {
    let selected = args.gpu.into_iter().collect::<Vec<_>>();
    hold_memory_on_gpus(args, selected)
}

fn hold_memory_on_gpus(args: HoldArgs, explicit_gpus: Vec<usize>) -> Result<()> {
    let total_target_bytes = parse_bytes(&args.memory)
        .with_context(|| format!("failed to parse --memory {}", args.memory))?;
    let reserve_bytes = parse_bytes(&args.reserve)
        .with_context(|| format!("failed to parse --reserve {}", args.reserve))?;
    let chunk_bytes = parse_bytes(&args.chunk_size)
        .with_context(|| format!("failed to parse --chunk-size {}", args.chunk_size))?;

    if total_target_bytes == 0 {
        bail!("--memory must be greater than 0");
    }
    if chunk_bytes == 0 {
        bail!("--chunk-size must be greater than 0");
    }

    let gpus = query_gpus()?;
    let selected = select_gpus(&gpus, &explicit_gpus, total_target_bytes, reserve_bytes)?;
    let per_gpu_targets = split_bytes_evenly(total_target_bytes, selected.len());

    for (gpu, per_gpu_target) in selected.iter().zip(&per_gpu_targets) {
        println!(
            "Selected GPU {} ({}) | total {} | used {} | free {} | target {}",
            gpu.index,
            gpu.name,
            format_mib(gpu.total_mib),
            format_mib(gpu.used_mib),
            format_mib(gpu.free_mib()),
            format_bytes(*per_gpu_target)
        );
    }
    println!(
        "Total target {} | reserve {} | chunk {} | status interval {}s",
        format_bytes(total_target_bytes),
        format_bytes(reserve_bytes),
        format_bytes(chunk_bytes),
        args.status_interval
    );

    let mut holds = Vec::new();
    for (gpu, per_gpu_target) in selected.into_iter().zip(per_gpu_targets) {
        let allocation = CudaAllocation::new(gpu.index)?;
        let (free_after_ctx, total_after_ctx) = allocation.memory_info()?;
        let usable = free_after_ctx.saturating_sub(reserve_bytes);
        if per_gpu_target > usable {
            bail!(
                "not enough free memory on GPU {} after context init: free {}, reserve {}, usable {}",
                gpu.index,
                format_bytes(free_after_ctx),
                format_bytes(reserve_bytes),
                format_bytes(usable)
            );
        }

        println!(
            "Context ready on GPU {} | CUDA sees free {} / total {}",
            gpu.index,
            format_bytes(free_after_ctx),
            format_bytes(total_after_ctx)
        );
        holds.push(ActiveHold {
            gpu,
            allocation,
            target_bytes: per_gpu_target,
        });
    }

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            stop.store(true, Ordering::SeqCst);
        })
        .context("failed to install Ctrl+C handler")?;
    }

    for hold in &mut holds {
        while hold.allocation.allocated_bytes < hold.target_bytes && !stop.load(Ordering::SeqCst) {
            let remaining = hold.target_bytes - hold.allocation.allocated_bytes;
            let next = remaining.min(chunk_bytes);
            hold.allocation.allocate(next)?;
            println!(
                "GPU {} allocated {} / {}",
                hold.gpu.index,
                format_bytes(hold.allocation.allocated_bytes),
                format_bytes(hold.target_bytes)
            );
        }
    }

    if stop.load(Ordering::SeqCst) {
        println!("Interrupted during allocation. Releasing all GPU allocations.");
        return Ok(());
    }

    println!(
        "Holding total {} across {} GPU(s). Press Ctrl+C to release.",
        format_bytes(total_target_bytes),
        holds.len(),
    );

    let interval = Duration::from_secs(args.status_interval.max(1));
    let selected_indexes = holds.iter().map(|hold| hold.gpu.index).collect::<Vec<_>>();
    while !stop.load(Ordering::SeqCst) {
        if let Err(err) = print_gpu_status_lines(&selected_indexes) {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            eprintln!("status update failed: {err}");
        }
        sleep_until_stopped(&stop, interval);
    }

    println!(
        "Received Ctrl+C. Releasing total {} from {} GPU(s).",
        format_bytes(total_target_bytes),
        holds.len()
    );
    Ok(())
}

fn select_gpus(
    gpus: &[GpuInfo],
    explicit_gpus: &[usize],
    total_target_bytes: u64,
    reserve_bytes: u64,
) -> Result<Vec<GpuInfo>> {
    if explicit_gpus.is_empty() {
        return select_auto_gpu(gpus, total_target_bytes, reserve_bytes).map(|gpu| vec![gpu]);
    }

    let per_gpu_targets = split_bytes_evenly(total_target_bytes, explicit_gpus.len());
    let mut selected = Vec::new();
    for (index, per_gpu_target) in explicit_gpus.iter().zip(per_gpu_targets) {
        let gpu = gpus
            .iter()
            .find(|gpu| gpu.index == *index)
            .cloned()
            .ok_or_else(|| anyhow!("GPU {} not found", index))?;
        ensure_gpu_has_capacity(&gpu, per_gpu_target, reserve_bytes)?;
        selected.push(gpu);
    }
    Ok(selected)
}

fn select_auto_gpu(gpus: &[GpuInfo], target_bytes: u64, reserve_bytes: u64) -> Result<GpuInfo> {
    let needed_mib = bytes_to_mib_ceil(target_bytes.saturating_add(reserve_bytes));
    gpus.iter()
        .filter(|gpu| gpu.free_mib() >= needed_mib)
        .max_by_key(|gpu| (gpu.free_mib(), Reverse(gpu.used_mib)))
        .cloned()
        .ok_or_else(|| auto_select_error(gpus, target_bytes, reserve_bytes))
}

fn ensure_gpu_has_capacity(gpu: &GpuInfo, target_bytes: u64, reserve_bytes: u64) -> Result<()> {
    let needed_mib = bytes_to_mib_ceil(target_bytes.saturating_add(reserve_bytes));
    if gpu.free_mib() < needed_mib {
        bail!(
            "GPU {} does not have enough free memory for target {} plus reserve {}. Current free memory: {}",
            gpu.index,
            format_bytes(target_bytes),
            format_bytes(reserve_bytes),
            format_mib(gpu.free_mib())
        );
    }
    Ok(())
}

fn split_bytes_evenly(total_bytes: u64, parts: usize) -> Vec<u64> {
    let parts = parts.max(1) as u64;
    let base = total_bytes / parts;
    let remainder = total_bytes % parts;
    (0..parts)
        .map(|index| base + u64::from(index < remainder))
        .collect()
}

fn auto_select_error(gpus: &[GpuInfo], target_bytes: u64, reserve_bytes: u64) -> anyhow::Error {
    let summary = gpus
        .iter()
        .map(|gpu| format!("GPU {} free {}", gpu.index, format_mib(gpu.free_mib())))
        .collect::<Vec<_>>()
        .join(", ");
    anyhow!(
        "no GPU has enough free memory for target {} plus reserve {}. Current free memory: {}",
        format_bytes(target_bytes),
        format_bytes(reserve_bytes),
        summary
    )
}

fn print_gpu_status_lines(indexes: &[usize]) -> Result<()> {
    let gpus = query_gpus()?;
    for index in indexes {
        let gpu = gpus
            .iter()
            .find(|gpu| gpu.index == *index)
            .ok_or_else(|| anyhow!("GPU {} disappeared from nvidia-smi output", index))?;
        println!(
            "[GPU {}] used {} / {} | free {} | util {}%",
            gpu.index,
            format_mib(gpu.used_mib),
            format_mib(gpu.total_mib),
            format_mib(gpu.free_mib()),
            gpu.util_pct
        );
    }
    Ok(())
}

fn print_gpu_table(gpus: &[GpuInfo]) {
    println!(
        "{:<5} {:<24} {:>10} {:>10} {:>10} {:>8}",
        "GPU", "Name", "Used", "Free", "Total", "Util"
    );
    for gpu in gpus {
        println!(
            "{:<5} {:<24} {:>10} {:>10} {:>10} {:>7}%",
            gpu.index,
            truncate_name(&gpu.name, 24),
            format_mib(gpu.used_mib),
            format_mib(gpu.free_mib()),
            format_mib(gpu.total_mib),
            gpu.util_pct
        );
    }
}

fn truncate_name(name: &str, width: usize) -> String {
    let chars = name.chars().collect::<Vec<_>>();
    if chars.len() <= width {
        return name.to_owned();
    }
    chars[..width.saturating_sub(3)].iter().collect::<String>() + "..."
}

fn query_gpus() -> Result<Vec<GpuInfo>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("failed to run nvidia-smi")?;

    if !output.status.success() {
        bail!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let stdout =
        String::from_utf8(output.stdout).context("nvidia-smi output was not valid UTF-8")?;
    let mut gpus = Vec::new();

    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let parts = line.split(',').map(|part| part.trim()).collect::<Vec<_>>();
        if parts.len() != 5 {
            bail!("unexpected nvidia-smi output line: {line}");
        }

        gpus.push(GpuInfo {
            index: parts[0]
                .parse()
                .with_context(|| format!("invalid GPU index in {line}"))?,
            name: parts[1].to_owned(),
            total_mib: parts[2]
                .parse()
                .with_context(|| format!("invalid total memory in {line}"))?,
            used_mib: parts[3]
                .parse()
                .with_context(|| format!("invalid used memory in {line}"))?,
            util_pct: parts[4]
                .parse()
                .with_context(|| format!("invalid utilization in {line}"))?,
        });
    }

    if gpus.is_empty() {
        bail!("nvidia-smi reported no GPUs");
    }

    Ok(gpus)
}

fn parse_bytes(input: &str) -> Result<u64> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("size cannot be empty");
    }

    let split_at = trimmed
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .unwrap_or(trimmed.len());
    let (value_raw, unit_raw) = trimmed.split_at(split_at);

    if value_raw.is_empty() {
        bail!("missing numeric value in size {trimmed}");
    }

    let value = value_raw
        .parse::<f64>()
        .with_context(|| format!("invalid numeric value in size {trimmed}"))?;
    if !value.is_finite() || value < 0.0 {
        bail!("invalid size value {trimmed}");
    }

    let multiplier = match unit_raw.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1f64,
        "k" | "kb" => 1_000f64,
        "m" | "mb" => 1_000_000f64,
        "g" | "gb" => 1_000_000_000f64,
        "t" | "tb" => 1_000_000_000_000f64,
        "ki" | "kib" => 1024f64,
        "mi" | "mib" => (1024u64.pow(2)) as f64,
        "gi" | "gib" => GIB as f64,
        "ti" | "tib" => (1024u64.pow(4)) as f64,
        other => bail!("unsupported size unit {other}"),
    };

    Ok((value * multiplier).round() as u64)
}

fn bytes_to_mib_ceil(bytes: u64) -> u64 {
    bytes.div_ceil(MIB)
}

fn format_mib(mib: u64) -> String {
    format_bytes(mib.saturating_mul(MIB))
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.0} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= 1024 {
        format!("{:.0} KiB", bytes as f64 / 1024f64)
    } else {
        format!("{bytes} B")
    }
}

fn sleep_until_stopped(stop: &AtomicBool, total: Duration) {
    let step = Duration::from_millis(100);
    let mut elapsed = Duration::ZERO;
    while elapsed < total && !stop.load(Ordering::SeqCst) {
        let current = (total - elapsed).min(step);
        thread::sleep(current);
        elapsed += current;
    }
}

unsafe fn cuda_check(code: CuResult, action: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }

    let mut ptr = std::ptr::null();
    let description = if unsafe { cuGetErrorString(code, &mut ptr) } == 0 && !ptr.is_null() {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    } else {
        "unknown CUDA error".to_owned()
    };

    Err(anyhow!(
        "{action} failed with CUDA error {code}: {description}"
    ))
}
