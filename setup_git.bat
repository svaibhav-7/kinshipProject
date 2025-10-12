@echo off
echo Setting up fresh Git repository for Kinship Verification Project...

cd /d "D:\SasiVaibhav\klu\3rd year\projects\kinship_verification\kinshipProject"

echo Removing existing nested Git repositories...
if exist "nvdiffrast\.git" (
    echo Removing nvdiffrast/.git directory...
    rmdir /s /q "nvdiffrast\.git"
)

if exist "Deep3DFaceRecon_pytorch\.git" (
    echo Removing Deep3DFaceRecon_pytorch/.git directory...
    rmdir /s /q "Deep3DFaceRecon_pytorch\.git"
)

echo Initializing new Git repository...
git init

echo Adding all files to staging...
git add .

echo Creating initial commit...
git commit -m "Initial commit: Kinship verification project with 3D face reconstruction

- Added comprehensive README with attribution
- Included security documentation and policies  
- Added proper LICENSE with third-party notices
- Complete citations for all dependencies
- Configured .gitignore for sensitive data protection

Features:
- 3D face reconstruction using Deep3DFaceRecon_pytorch
- Siamese neural networks for kinship verification
- Comprehensive pipelines for feature extraction
- Security-first approach with data protection"

echo.
echo Git repository initialized successfully!
echo.
echo Next steps:
echo 1. Create a new repository on GitHub
echo 2. Add remote: git remote add origin https://github.com/yourusername/kinship-verification.git
echo 3. Push to GitHub: git push -u origin main
echo.
echo IMPORTANT: Review SECURITY_CHECKLIST.md before pushing to GitHub!
echo.
pause