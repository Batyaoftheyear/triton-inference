$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResultsDir = Join-Path $ProjectRoot "results"
$ImageName = "triton-mobilenet"

New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

docker run --rm `
  -v "${ProjectRoot}\analyzer:/workspace/analyzer" `
  -v "${ResultsDir}:/workspace/results" `
  $ImageName `
  bash -lc "pip install --no-cache-dir protobuf==4.25.1 triton-model-analyzer==1.37.0 && model-analyzer profile --config-file /workspace/analyzer/config.yaml --triton-launch-mode=local --export-path /workspace/results/model_analyzer | tee /workspace/results/model_analyzer.txt"
