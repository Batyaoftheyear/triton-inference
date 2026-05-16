$ModelName = "mobilenet"
$ServerUrl = "host.docker.internal:8000"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResultsDir = Join-Path $ProjectRoot "results"

New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

$PerfCommand = "perf_analyzer -m $ModelName -u $ServerUrl --input-data random --shape IMAGE:3,224,224 | tee /workspace/results/perf_analyzer.txt"

docker run --rm `
  -v "${ResultsDir}:/workspace/results" `
  nvcr.io/nvidia/tritonserver:23.12-py3-sdk `
  bash -lc $PerfCommand
