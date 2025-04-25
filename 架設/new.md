# CMO

## 初始設置

### 1. pycmopen clone
連結:[THESE](https://forgejo.taiyopen.com/Taiyopen/pycmopen/src/branch/main)
(路徑:`C:\Users\yuhsu\OneDrive\桌面\pycmopen`)

### 2. conda創建環境(name:cmo)
1. `conda create -n fcmo python=3.10` : 創建環境(name:fcmo)
2. `conda activate fcmo` : 開啟環境
3. `cd C:\Users\yuhsu\OneDrive\桌面\pycmopen`
4. `pip install -e .` : 安裝
5. 安裝cuda及pytorch:`pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124`

### 3. 架設cmo場景
1. 改`init.lua`檔[路徑]()
8. `cd "C:\Users\yuhsu\pycmo-main\scripts\steam_demo"` :試跑 step1
9. `python demo.py` :試跑 step2
