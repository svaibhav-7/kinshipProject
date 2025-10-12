# PowerShell script to initialize Git repository for Kinship Verification Project

Write-Host "Setting up fresh Git repository for Kinship Verification Project..." -ForegroundColor Green

# Change to project directory
$projectDir = "D:\SasiVaibhav\klu\3rd year\projects\kinship_verification\kinshipProject"
Set-Location $projectDir

# Remove nested Git repositories
Write-Host "Checking for nested Git repositories..." -ForegroundColor Yellow

if (Test-Path "nvdiffrast\.git") {
    Write-Host "Removing nvdiffrast/.git directory..." -ForegroundColor Yellow
    Remove-Item -Path "nvdiffrast\.git" -Recurse -Force
}

if (Test-Path "Deep3DFaceRecon_pytorch\.git") {
    Write-Host "Removing Deep3DFaceRecon_pytorch/.git directory..." -ForegroundColor Yellow
    Remove-Item -Path "Deep3DFaceRecon_pytorch\.git" -Recurse -Force
}

# Check if main .git already exists and remove it for fresh start
if (Test-Path ".git") {
    Write-Host "Removing existing .git directory for fresh start..." -ForegroundColor Yellow
    Remove-Item -Path ".git" -Recurse -Force
}

# Initialize new Git repository
Write-Host "Initializing new Git repository..." -ForegroundColor Green
git init

# Set up Git configuration (optional - user should set their own)
Write-Host "Setting up Git configuration..." -ForegroundColor Green
Write-Host "Please configure your Git username and email:"
Write-Host "git config user.name 'Your Name'"
Write-Host "git config user.email 'your.email@example.com'"

# Add all files
Write-Host "Adding files to staging area..." -ForegroundColor Green
git add .

# Create initial commit
Write-Host "Creating initial commit..." -ForegroundColor Green
$commitMessage = @"
Initial commit: Kinship verification project with 3D face reconstruction

- Added comprehensive README with attribution
- Included security documentation and policies  
- Added proper LICENSE with third-party notices
- Complete citations for all dependencies
- Configured .gitignore for sensitive data protection

Features:
- 3D face reconstruction using Deep3DFaceRecon_pytorch
- Siamese neural networks for kinship verification
- Comprehensive pipelines for feature extraction
- Security-first approach with data protection
"@

git commit -m $commitMessage

Write-Host "`nGit repository initialized successfully!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Create a new repository on GitHub" -ForegroundColor White
Write-Host "2. Add remote: git remote add origin https://github.com/yourusername/kinship-verification.git" -ForegroundColor White
Write-Host "3. Push to GitHub: git push -u origin main" -ForegroundColor White
Write-Host "`nIMPORTANT: Review SECURITY_CHECKLIST.md before pushing to GitHub!" -ForegroundColor Red

# Check repository status
Write-Host "`nRepository Status:" -ForegroundColor Cyan
git status
git log --oneline -1

Write-Host "`nPress any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")