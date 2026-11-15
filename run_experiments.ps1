# Run Emmanuel-NS experiments 1 to 10, one after another.
# Close this window, or press Ctrl+C, to stop. The current run still saves.
# Pass -Episodes and -StartAt. A run that already reached that episode count is skipped.
# A partial run that already has a log is kept and is not started over.

param(
    [int]$Episodes = 400,
    [int]$StartAt = 1
)

# Continue, not Stop: the Atari library writes its banner on stderr, and
# Stop would treat that banner as a failed command.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:Path = "$env:USERPROFILE\.local\bin;" + $env:Path
$env:PYTHONUNBUFFERED = "1"

$member = "Emmanuel-NS"
$root = Join-Path $PSScriptRoot "models\$member"
New-Item -ItemType Directory -Force -Path $root | Out-Null

$already = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "train\.py" }
if ($already) {
    Write-Output "train.py is already running. Stop that window before starting another sweep."
    exit 1
}

# Ask Windows to stay awake while this window is open. Closing it clears the request.
# This does not change the power plan.
Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

$experiments = @(
    @{ n = 1;  lr = "0.0001";  gamma = "0.99";  batch = 32;  eps_start = "1.0"; eps_end = "0.05"; eps_decay = "0.992" },
    @{ n = 2;  lr = "0.001";   gamma = "0.99";  batch = 32;  eps_start = "1.0"; eps_end = "0.05"; eps_decay = "0.992" },
    @{ n = 3;  lr = "0.00001"; gamma = "0.99";  batch = 32;  eps_start = "1.0"; eps_end = "0.05"; eps_decay = "0.992" },
    @{ n = 4;  lr = "0.0001";  gamma = "0.9";   batch = 32;  eps_start = "1.0"; eps_end = "0.05"; eps_decay = "0.992" },
    @{ n = 5;  lr = "0.0001";  gamma = "0.999"; batch = 32;  eps_start = "1.0"; eps_end = "0.05"; eps_decay = "0.992" },
    @{ n = 6;  lr = "0.0001";  gamma = "0.99";  batch = 16;  eps_start = "1.0"; eps_end = "0.05"; eps_decay = "0.992" },
    @{ n = 7;  lr = "0.0001";  gamma = "0.99";  batch = 128; eps_start = "1.0"; eps_end = "0.05"; eps_decay = "0.992" },
    @{ n = 8;  lr = "0.0001";  gamma = "0.99";  batch = 32;  eps_start = "0.2"; eps_end = "0.05"; eps_decay = "0.992" },
    @{ n = 9;  lr = "0.0001";  gamma = "0.99";  batch = 32;  eps_start = "1.0"; eps_end = "0.4";  eps_decay = "0.992" },
    @{ n = 10; lr = "0.0001";  gamma = "0.99";  batch = 32;  eps_start = "1.0"; eps_end = "0.05"; eps_decay = "0.97" }
)

try {
    foreach ($experiment in $experiments) {
        $folder = Join-Path $root ("experiment_{0:d2}" -f $experiment.n)
        $summaryPath = Join-Path $folder "summary.json"
        if ($experiment.n -lt $StartAt) {
            Write-Output "Skipping experiment $($experiment.n): start point is experiment $StartAt."
            continue
        }
        $csvPath = Join-Path $folder "training_log.csv"
        if (Test-Path $summaryPath) {
            $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
            if ($summary.episodes_completed -ge $Episodes) {
                Write-Output "Skipping experiment $($experiment.n): already finished $($summary.episodes_completed) episodes."
                continue
            }
            if (Test-Path $csvPath) {
                Write-Output "Skipping experiment $($experiment.n): partial run kept ($($summary.episodes_completed) episodes). It is not started over."
                continue
            }
        }

        $log = Join-Path $root ("experiment_{0:d2}.log" -f $experiment.n)
        Write-Output ""
        Write-Output "=== Experiment $($experiment.n) / 10  ($Episodes episodes) ==="
        Write-Output "Log: $log"

        # Overwrite the log. An unfinished experiment starts again from episode 1.
        Set-Content -Path $log -Value "experiment $($experiment.n) started $(Get-Date -Format o)"
        & uv run python -u train.py `
            --member $member `
            --experiment $experiment.n `
            --episodes $Episodes `
            --learning-rate $experiment.lr `
            --gamma $experiment.gamma `
            --batch-size $experiment.batch `
            --epsilon-start $experiment.eps_start `
            --epsilon-end $experiment.eps_end `
            --epsilon-decay $experiment.eps_decay | ForEach-Object {
                Write-Output $_
                Add-Content -Path $log -Value $_
            }

        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $($experiment.n) failed with exit code $LASTEXITCODE"
        }
    }

    Write-Output ""
    Write-Output "All planned experiments finished."
    Write-Output "Fill Emmanuel-NSHyperparameter_experiment_results.md from each summary.json."
    Write-Output "Then promote the best one, for example:"
    Write-Output "  uv run python train.py --member $member --promote-from 1"
}
finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}
