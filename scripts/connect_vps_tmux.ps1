param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("neural_codex_auth", "neural_whatsapp_pair")]
    [string]$Session
)

$envPath = "C:\Users\graci\birdclef\.env"
if (-not (Test-Path -LiteralPath $envPath)) {
    throw "Missing $envPath"
}

$line = Get-Content -LiteralPath $envPath |
    Where-Object { $_ -match '^DROPLET_IP=' } |
    Select-Object -First 1

if (-not $line) {
    throw "DROPLET_IP not found in $envPath"
}

$ip = ($line -replace '^DROPLET_IP=', '').Trim().Trim('"').Trim("'")
ssh -t "root@$ip" "tmux attach -t $Session"

