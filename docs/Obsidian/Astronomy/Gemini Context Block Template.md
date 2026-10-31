### The Anatomy of an Effective Context Block

An effective context block covers **five simple fields**:

1. **Machine & Hardware:** OS, GPU, storage layout, and role (e.g., Mac Studio for stacking, Ubuntu "Inti" for PyTorch).
    
2. **Current Target & Data:** Scope, exposure settings, mount type, and framing.
    
3. **Core Constraints:** Specific boundaries (e.g., keep Inti's local display functional; headless via SSH/tmux/Jupyter; avoid CLI package breaks).
    
4. **Active Track / Sub-Project:** Which part of the project this specific chat owns.
    
5. **Next Immediate Action:** The exact command you just ran or are about to run.
    

### A Reusable Template for Obsidian or Your Notes

Save this template directly in your **Obsidian** logbook:

Markdown

```
### SYSTEM CONTEXT BLOCK: [TRACK NAME]
- Hardware/Host: [e.g., Machine "Inti", Ubuntu 22.04 LTS, RTX 2080, Conda]
- Input Data: [e.g., Master 16-bit TIFF from DWARF III / M31]
- Environment/Tools: [e.g., PyTorch, StarNet CLI, JupyterLab]
- Constraints: [e.g., Preserve desktop Xorg display for tutoring; headless SSH workflow]
- Active Goal: [e.g., Set up PyTorch inference pipeline for starless target upscaling]
- Current Immediate Step: [e.g., Reviewing nvidia-smi output below]
```

### How to Use It

- **To resume here:** Just drop your terminal outputs directly in this chat whenever you have them.
    
- **To start a new chat (Track 2):** Open a fresh chat, paste that text block with your `nvidia-smi` and `ubuntu-drivers devices`output in message #1, and the new session will pick up immediately without missing a beat.