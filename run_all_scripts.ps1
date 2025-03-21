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
        Args = @("--RefreshMode")
    },
    @{
        Name = "Indicators Script"
        File = "3__Indicators.py"
        Args = @()
    },
    @{
        Name = "Predictor Script"
        File = "4__Predictor.py"
        Args = @("--predict", "--model", "xgb")
    },
    @{
        Name = "Nightly BackTester Script"
        File = "5__NightlyBackTester.py"
        Args = @("--force")
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

# Find and read the Buy Signals parquet file
try {
    Write-Log "Looking for Buy Signals parquet file..." -Color Cyan
    $buySignalsFile = Get-ChildItem -Path "$basePath\Data" -Recurse -Filter "*Buy*Signals.parquet" | Select-Object -First 1

    if ($buySignalsFile) {
        Write-Log "Found Buy Signals file: $($buySignalsFile.FullName)" -Color Green
        
        # Create a Python script to read the parquet file
        $tempPythonScript = [System.IO.Path]::GetTempFileName() + ".py"
        @"
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Get the next trading day (simple approximation)
today = datetime.now()
next_trading_day = today + timedelta(days=1)
if next_trading_day.weekday() >= 5:  # Saturday or Sunday
    next_trading_day = today + timedelta(days=(7 - today.weekday()))

try:
    # Read the parquet file
    file_path = '$($buySignalsFile.FullName.Replace("\", "\\"))'
    df = pd.read_parquet(file_path)
    
    # Check for any buy signals where IsCurrentlyBought is True
    buy_signals = df[df['IsCurrentlyBought'] == True].copy()
    
    if not buy_signals.empty:
        print("\n===== BUY SIGNALS FOR NEXT TRADING DAY =====")
        print(f"Next trading day: {next_trading_day.strftime('%Y-%m-%d')}")
        print(f"Number of active buy signals: {len(buy_signals)}")
        print("\nTop 5 signals by UpProbability:")
        
        # Sort by UpProbability and display top 5
        top_signals = buy_signals.sort_values('UpProbability', ascending=False).head(5)
        for i, row in top_signals.iterrows():
            print(f"Symbol: {row['Symbol']}, UpProbability: {row['UpProbability']:.4f}, Buy Price: {row['LastBuySignalPrice']:.2f}, Position Size: {row['PositionSize']}")
        
        print("\n===========================================")
    else:
        print("\nNo active buy signals found for the next trading day.")
    
except Exception as e:
    print(f"Error reading parquet file: {str(e)}")
"@ | Out-File -FilePath $tempPythonScript -Encoding utf8

        # Execute the Python script
        $pinfo = New-Object System.Diagnostics.ProcessStartInfo
        $pinfo.FileName = $pythonExe
        $pinfo.Arguments = $tempPythonScript
        $pinfo.UseShellExecute = $false
        $pinfo.RedirectStandardOutput = $true
        $pinfo.CreateNoWindow = $true

        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $pinfo
        $process.Start() | Out-Null
        $output = $process.StandardOutput.ReadToEnd()
        $process.WaitForExit()

        # Display the output
        Write-Host $output
        Write-Log $output -Color Yellow
        
        # Clean up the temporary Python script
        Remove-Item -Path $tempPythonScript -Force
    } else {
        Write-Log "No Buy Signals parquet file found." -Color Yellow
    }
}
catch {
    Write-Log "Error while attempting to read Buy Signals file: $($_.Exception.Message)" -Color Red
}

Write-Log "This Day's Buy Signals Saved to trading_data.parquet" -Color Yellow
Write-Log "Script execution completed." -Color Yellow
Write-Log "=========================================" -Color White

# Keep the window open for one hour
Write-Host "`nKeeping window open for 1 hour to review results..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to exit earlier if needed." -ForegroundColor Yellow

try {
    $endTime = (Get-Date).AddHours(1)
    while ((Get-Date) -lt $endTime) {
        Start-Sleep -Seconds 60
        $timeLeft = [math]::Round(($endTime - (Get-Date)).TotalMinutes, 0)
        Write-Host "`rTime remaining: $timeLeft minutes     " -NoNewline -ForegroundColor Gray
    }
}
catch {
    # Just exit if there's an error or user interruption
}

# Automatically exit after completion
exit 0