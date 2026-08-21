# HH Goa 2026 Task #2: Windows PowerShell Bootstrap

Write-Host "=== HH Goa 2026 Task #2: Deploying Stack ===" -ForegroundColor Green

# 1. Build
Write-Host "1. Building Docker Images..." -ForegroundColor Cyan
docker compose build

# 2. Spin up DBs
Write-Host "2. Bootstrapping Qdrant & Redis..." -ForegroundColor Cyan
docker compose up -d qdrant redis

# 3. Wait for Qdrant
Write-Host "3. Waiting for Qdrant Vector DB to initialize..." -ForegroundColor Cyan
while ($true) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method Get -TimeoutSec 1 -ErrorAction Stop
        if ($response) {
            Write-Host "Qdrant is ready." -ForegroundColor Green
            break
        }
    }
    catch {
        Write-Host "Waiting for Qdrant to start..."
        Start-Sleep -Seconds 2
    }
}

# 4. Ingest Check
Write-Host "4. Checking if dataset index exists..." -ForegroundColor Cyan
$indexExists = $false
try {
    $colResponse = Invoke-RestMethod -Uri "http://localhost:6333/collections/msmarco_chunks" -Method Get -ErrorAction Stop
    if ($colResponse.result.status -eq "ok" -or $colResponse.result.vectors_count -gt 0) {
        $indexExists = $true
    }
}
catch {}

if ($indexExists) {
    Write-Host "Vector index 'msmarco_chunks' already exists. Skipping ingestion." -ForegroundColor Yellow
}
else {
    Write-Host "Vector index not found. Running dataset ingestion script..." -ForegroundColor Cyan
    docker compose run --entrypoint "python backend/scripts/ingest.py 30 hi" backend
}

# 5. Start Servers
Write-Host "5. Spin up Frontend and Backend web servers..." -ForegroundColor Cyan
docker compose up -d

# 6. Health validation
Write-Host "6. Running health check..." -ForegroundColor Cyan
Start-Sleep -Seconds 5
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get -ErrorAction Stop
    if ($health.status -eq "healthy") {
        Write-Host "`n=== SUCCESS: Stack is healthy and running! ===" -ForegroundColor Green
        Write-Host "Frontend: http://localhost:3000" -ForegroundColor Green
        Write-Host "Backend API: http://localhost:8000" -ForegroundColor Green
        Write-Host "Health status: " -NoNewline
        Write-Object $health.details
    }
}
catch {
    Write-Host "`n=== WARNING: Health check failed or returned unhealthy! Run 'docker compose logs' to inspect ===" -ForegroundColor Red
}
