# CMO

## 初始設置

### 1. pycmopen clone
連結:[THESE](https://forgejo.taiyopen.com/Taiyopen/pycmopen/src/branch/main)
(放置路徑:`C:\Users\yuhsu\OneDrive\桌面\pycmopen`)
(放哪都行)

### 2. conda創建環境(name:cmo)
1. `conda create -n fcmo python=3.10` : 創建環境(name:`fcmo`)
2. `conda activate fcmo` : 開啟環境
3. `cd C:\Users\yuhsu\OneDrive\桌面\pycmopen`
4. `pip install -e .` : 安裝
5. 安裝cuda及pytorch:`pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124` (4040系顯卡)

### 3. 架設cmo場景(以feudal為例)
1. 將`C:\Users\yuhsu\OneDrive\桌面\pycmopen\pycmo\configs\config.py`的`config_template.py`改名為`config.py`，並更新內容:[路徑](https://github.com/Yuu-Hsuan/CMO/blob/main/%E6%9E%B6%E8%A8%AD/init.py)
2. 更新`C:\Users\yuhsu\OneDrive\桌面\pycmopen\scripts\FeUdal\init.lua`的內容:[路徑](https://github.com/Yuu-Hsuan/CMO/blob/main/%E6%9E%B6%E8%A8%AD/init.lua)
3. 開啟遊戲，點選`create new Scenario`
4. 點擊左上角的`file`選擇`load`選擇`browse`
5. 選擇`C:\Users\yuhsu\OneDrive\桌面\pycmopen\scen\MyScenario1.scen`(目前feudal在跑的場景路徑插入)
6. 選擇`enter scenario`
7. 選擇上方欄位的`editor`
8. 選擇`lua script console`
9. 將`C:\Users\yuhsu\OneDrive\桌面\pycmopen\scripts\FeUdal\init.lua`貼到下方白色欄位，點擊`run`
10. 至`C:\Program Files (x86)\Steam\steamapps\common\Command - Modern Operations\ImportExport`(遊戲儲存位置)觀察是否儲存成功，滑到最下面，會有三個檔
    * MyScenario1.inst
    * MyScenario1  (XML檔)
    * MyScenario1_scen_has_ended.inst

### 4.執行feudal的`MyScenario1`環境
1. 打開遊戲(小視窗左上角打叉) 、 自行切換為英文輸入法
2. 進入環境輸入: `python .\scripts\Feudal\demo.py --config=scripts/Feudal/config.yaml`
