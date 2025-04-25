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

### 3. 架設cmo場景(以feudal為例)
1. 將`C:\Users\yuhsu\OneDrive\桌面\pycmopen\pycmo\configs\config.py`的`config_template.py`改名為`config.py`，並更新內容:[路徑](https://github.com/Yuu-Hsuan/CMO/blob/main/%E6%9E%B6%E8%A8%AD/init.py)
2. 更新`C:\Users\yuhsu\OneDrive\桌面\pycmopen\scripts\FeUdal\init.lua`的內容
8. `cd "C:\Users\yuhsu\pycmo-main\scripts\steam_demo"` :試跑 step1
9. `python demo.py` :試跑 step2
