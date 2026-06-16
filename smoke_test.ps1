$py    = "$env:USERPROFILE\AppData\Local\Programs\Python\Launcher\py.exe"
$ROOT  = "E:\trabalho-cc"
$PASS  = 0
$FAIL  = 0

Set-Location $ROOT

function Run-Check($label, $script) {
    $out = & $py -3.11 $script 2>&1
    $out | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -eq 0) { $script:PASS++ }
    else                     { $script:FAIL++ }
}

function Section($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

# ------------------------------------------------------------------
Section "1. PIPELINE COMPLETO (main.py)"
# ------------------------------------------------------------------
$out = & $py -3.11 main.py 2>&1
$out | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [PASS] main.py concluiu sem erros" -ForegroundColor Green
    $PASS++
} else {
    Write-Host "  [FAIL] main.py retornou erro" -ForegroundColor Red
    $FAIL++
}

# ------------------------------------------------------------------
Section "2. ARTEFATOS GERADOS"
# ------------------------------------------------------------------
$artefatos = @(
    "data\processed\f1_processed.parquet",
    "data\processed\f1_race_results.parquet",
    "data\final\f1_dataset_final.parquet",
    "models\modelo_top10.pkl",
    "mlflow.db",
    "experiments\avaliacao.txt"
)
foreach ($f in $artefatos) {
    if (Test-Path "$ROOT\$f") {
        Write-Host "  [PASS] Existe: $f" -ForegroundColor Green
        $PASS++
    } else {
        Write-Host "  [FAIL] FALTANDO: $f" -ForegroundColor Red
        $FAIL++
    }
}

# ------------------------------------------------------------------
Section "3. SHAPES DOS DATASETS"
# ------------------------------------------------------------------
Run-Check "shapes" "tests\check_shapes.py"

# ------------------------------------------------------------------
Section "4. TARGET BALANCEADO"
# ------------------------------------------------------------------
Run-Check "target" "tests\check_target.py"

# ------------------------------------------------------------------
Section "5. SEM LEAKAGE NO DATASET FINAL"
# ------------------------------------------------------------------
Run-Check "leakage" "tests\check_leakage.py"

# ------------------------------------------------------------------
Section "6. MODELO CARREGA E PREDIZ"
# ------------------------------------------------------------------
Run-Check "model" "tests\check_model.py"

# ------------------------------------------------------------------
Section "7. API -- ENDPOINTS"
# ------------------------------------------------------------------
Write-Host "  Iniciando uvicorn em background (porta 8000)..."
$uvicorn = Start-Process -FilePath $py `
    -ArgumentList "-3.11 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000" `
    -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 5

Run-Check "api" "tests\check_api.py"

Stop-Process -Id $uvicorn.Id -Force -ErrorAction SilentlyContinue
Write-Host "  uvicorn encerrado."

# ------------------------------------------------------------------
Section "RESULTADO FINAL"
# ------------------------------------------------------------------
Write-Host ""
$total = $PASS + $FAIL
if ($FAIL -eq 0) {
    Write-Host "  TUDO OK -- $PASS/$total testes passaram." -ForegroundColor Green
} else {
    Write-Host "  $FAIL FALHA(S) -- $PASS/$total testes passaram." -ForegroundColor Red
}
Write-Host ""
