# Automated Deploy Script for RunPod Serverless Workers
# Run this script to build and push your Docker containers to Docker Hub automatically.

$ErrorActionPreference = "Stop"

Clear-Host
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   AI Studio - RunPod Serverless Deployer         " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if Docker is running
try {
    & docker ps > $null
} catch {
    Write-Host "[!] Docker is not running. Please open Docker Desktop and try again." -ForegroundColor Red
    Exit
}

# 2. Get Docker Hub username
$dockerUser = Read-Host "Enter your Docker Hub username (e.g., Unknwnusr22)"
if ([string]::IsNullOrWhiteSpace($dockerUser)) {
    Write-Host "[!] Docker Hub username cannot be empty." -ForegroundColor Red
    Exit
}

# 3. Choose which worker to deploy
Write-Host ""
Write-Host "Which worker(s) would you like to build and push?"
Write-Host "1) Image-to-Video (LTX 2.3 / 10Eros)"
Write-Host "2) Image-to-Image (Flux.2 Klein)"
Write-Host "3) Both"
$choice = Read-Host "Select option (1, 2, or 3)"

# Ensure directory is root
$root = Get-Item $PSScriptRoot

if ($choice -eq "1" -or $choice -eq "3") {
    Write-Host ""
    Write-Host "--------------------------------------------------" -ForegroundColor Yellow
    Write-Host "Building and Pushing: Image-to-Video Worker (LTX 2.3)" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------" -ForegroundColor Yellow
    
    $i2vPath = Join-Path $root.FullName "workers\i2v"
    $tag = "$($dockerUser.ToLower())/po-i2v:latest"
    
    Write-Host "Building Docker Image: $tag..." -ForegroundColor Cyan
    & docker build -t $tag $i2vPath
    
    Write-Host "Pushing Docker Image to Docker Hub..." -ForegroundColor Cyan
    & docker push $tag
    
    Write-Host "[✓] LTX 2.3 Image-to-Video worker deployed successfully!" -ForegroundColor Green
    Write-Host "Your Container Image URL: $tag" -ForegroundColor Green
}

if ($choice -eq "2" -or $choice -eq "3") {
    Write-Host ""
    Write-Host "--------------------------------------------------" -ForegroundColor Yellow
    Write-Host "Building and Pushing: Image-to-Image Worker (Flux.2)" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------" -ForegroundColor Yellow
    
    $i2iPath = Join-Path $root.FullName "workers\i2i"
    $tag = "$($dockerUser.ToLower())/po-i2i:latest"
    
    Write-Host "Building Docker Image: $tag..." -ForegroundColor Cyan
    & docker build -t $tag $i2iPath
    
    Write-Host "Pushing Docker Image to Docker Hub..." -ForegroundColor Cyan
    & docker push $tag
    
    Write-Host "[✓] Flux.2 Image-to-Image worker deployed successfully!" -ForegroundColor Green
    Write-Host "Your Container Image URL: $tag" -ForegroundColor Green
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "                  ALL DONE!                       " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "You can now go to RunPod and use the container URLs shown above." -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit..."
