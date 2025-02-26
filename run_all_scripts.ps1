# Define the base path where your Python scripts are located
$basePath = "C:\Users\Masam\Desktop\Stock-Market-LSTM"

# Define the log file path
$logFile = "$basePath\Data\logging\__run_all_scripts.log"

# Define the path to the Python executable
$pythonExe = "C:\Users\Masam\miniconda3\envs\tf\python.exe"

# Define the scripts to run with their respective arguments
$scripts = @(
    @{
        Name = "Ticker Downloader"
        File = "1__TickerDownloader.py"
        Args = @("--ImmediateDownload")
    },
    @{
        Name = "Bulk Price Downloader"
        File = "2__BulkPriceDownloader.py"
        Args = @("--RefreshMode", "--skip-minute-data", "--request-delay", "0.5", "--batch-pause", "3")
    },
    @{
        Name = "Indicators Script"
        File = "3__Indicators.py"
        Args = @()
    },
    @{
        Name = "Predictor Script"
        File = "4__Predictor.py"
        Args = @("--predict")
    },
    @{
        Name = "Nightly Broker Script"
        File = "5__NightlyBroker.py"
        Args = @("--RunStocks", "1")
    }
)

# Function to write log messages
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("White", "Gray", "Red", "Green", "Yellow", "Cyan")]
        [string]$Color = "White"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage -ForegroundColor $Color
    Add-Content -Path $logFile -Value $logMessage
}

# Ensure the logging directory exists
$logDir = Split-Path $logFile -Parent
if (-not (Test-Path $logDir)) {
    try {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        Write-Log "Created logging directory at $logDir" -Color Green
    }
    catch {
        Write-Host "Failed to create logging directory at $logDir. Exiting script." -ForegroundColor Red
        exit 1
    }
}

# Clear the host screen and display startup messages
Clear-Host
Write-Host "Starting the execution of Python scripts..." -ForegroundColor Cyan
Write-Host "Logs will be saved to $logFile" -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Start logging
Write-Log "=========================================" -Color White
Write-Log "Script execution started. Base path: $basePath" -Color White
Write-Log "=========================================" -Color White

foreach ($script in $scripts) {
    Write-Log "Preparing to run '$($script.Name)'..." -Color Cyan
    Start-Sleep -Seconds 30

    $timer = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        $scriptPath = Join-Path -Path $basePath -ChildPath $script.File

        if (-not (Test-Path $scriptPath)) {
            throw "Script file not found: $scriptPath"
        }

        if (-not (Test-Path $pythonExe)) {
            throw "Python executable not found at $pythonExe"
        }

        $argList = @($scriptPath) + $script.Args
        $command = "$pythonExe " + ($argList -join ' ')
        Write-Log "Executing: $command" -Color Gray

        $pinfo = New-Object System.Diagnostics.ProcessStartInfo
        $pinfo.FileName = $pythonExe
        $pinfo.Arguments = ($argList -join ' ')
        $pinfo.UseShellExecute = $false
        $pinfo.CreateNoWindow = $true

        # No redirection of stdout/stderr
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $pinfo
        $process.Start() | Out-Null

        # Wait for the process to exit before continuing
        $process.WaitForExit()

        # Now that the process has exited, we can check the exit code
        if ($process.ExitCode -ne 0) {
            throw "Script '$($script.Name)' exited with code $($process.ExitCode)"
        }

        Write-Log "'$($script.Name)' completed successfully." -Color Green
    }
    catch {
        Write-Log "Error running '$($script.Name)'. Error details:" -Color Red
        Write-Log $_.Exception.Message -Color Red
    }
    finally {
        $timer.Stop()
        $elapsed = [math]::Round($timer.Elapsed.TotalSeconds, 2)
        Write-Log "Time elapsed for '$($script.Name)': $elapsed seconds." -Color Cyan
        Write-Log "-----------------------------------------" -Color White
    }
}

Write-Log "All scripts have been executed." -Color Yellow
Write-Log "This Days's Buy Signals Saved to trading_data.parquet" -Color Yellow
Write-Log "Script execution completed." -Color Yellow
Write-Log "=========================================" -Color White

# Automatically exit after completion
exit 0