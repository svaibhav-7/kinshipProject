# Git Repository Setup Guide

## Problem: Existing Git Connections

You mentioned the repository seems already connected. This is likely due to:

1. **Nested Git repositories** in subdirectories (nvdiffrast, Deep3DFaceRecon_pytorch)
2. **Cloned repositories** that still have their original remote connections
3. **Hidden .git directories** in subfolders

## Solution: Fresh Git Repository

### Option 1: Use the Automated Script (Recommended)

Run one of these scripts to automatically clean and initialize:

**For Windows Command Prompt:**
```cmd
setup_git.bat
```

**For PowerShell:**
```powershell
.\setup_git.ps1
```

### Option 2: Manual Setup

1. **Remove nested Git repositories:**
```cmd
rmdir /s /q nvdiffrast\.git
rmdir /s /q Deep3DFaceRecon_pytorch\.git
```

2. **Initialize fresh Git repository:**
```cmd
git init
git add .
git commit -m "Initial commit: Kinship verification project"
```

3. **Add your GitHub remote:**
```cmd
git remote add origin https://github.com/yourusername/your-repo-name.git
git branch -M main
git push -u origin main
```

## Pre-Push Security Checklist

**⚠️ CRITICAL: Complete the security checklist before pushing!**

1. Review `SECURITY_CHECKLIST.md`
2. Ensure no sensitive data in commits
3. Verify all attributions are correct
4. Check .gitignore excludes large files

## Recommended Repository Name

Use a professional name like:
- `kinship-verification-3d`
- `facial-kinship-detection`
- `3d-kinship-verification`

## GitHub Repository Settings

When creating on GitHub:

1. **Repository name:** Choose from above suggestions
2. **Description:** "Kinship verification using 3D face reconstruction and Siamese networks"
3. **Visibility:** Public (since it's research code with proper attribution)
4. **Initialize:** Don't add README/LICENSE/gitignore (you already have them)

## After Pushing

1. Add repository topics on GitHub:
   - `kinship-verification`
   - `3d-face-reconstruction` 
   - `siamese-networks`
   - `computer-vision`
   - `deep-learning`
   - `pytorch`

2. Update the README.md with the correct repository URL

3. Consider creating releases for major versions

## Troubleshooting

**If you still see unwanted connections:**

1. Check for hidden .git directories:
```cmd
dir /ah /s .git
```

2. Remove any found .git directories in subdirectories

3. Start fresh with `git init`

**If large files cause issues:**
- Use Git LFS for model files
- Ensure .gitignore excludes them first
- Consider hosting models separately

## Security Reminder

🔐 This project handles facial biometric data. Ensure:
- No personal test images in commits
- Proper dataset licensing compliance
- Privacy considerations documented
- Ethical use guidelines followed