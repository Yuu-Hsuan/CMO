# CMO

## 初始設置

### 1. pycmo 改寫
連結:[THESE](https://github.com/cjy202411/IRIS-CMO/blob/main/HowToSeries/CMO/CMO%E7%92%B0%E5%A2%83%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8_v0.pdf)

### 2. conda創建環境(name:cmo)
1. `conda create -n cmo python=3.10` : 創建環境(name:cmo)
2. `conda activate cmo` : 開啟環境
3. `cd C:\Users\yuhsu\pycmo-main` 
4. `pip install .` : 安裝[pycmo](https://github.com/duyminh1998/pycmo)
5. `cd "C:\Users\yuhsu\pycmo-main\scripts\steam_demo"` :試跑 step1
6. `python demo.py` :試跑 step2

## 檔案概述
### manager類別
1. `C:\Users\yuhsu\pycmo-main\scripts\steam_demo\demo.py` :

   腳本的主檔案，負責啟動並執行情境。它包含了控制情境運行的邏輯，例如設置單位、更新指令、執行回合等。(示範腳本)

### worker類別
1. `C:\Users\yuhsu\anaconda3\envs\cmo\lib\site-packages\pycmo\env\cmo_env.py` :

   定義環境物件，負責與外部情境進行交互。檔案中的 `step` 函數會向情境發送命令，並檢查遊戲是否結束。如果遊戲還未結束，該檔案會繼續發送命令或重啟情境。
   
2. `C:\Users\yuhsu\anaconda3\envs\cmo\Lib\site-packages\pycmo\lib\run_loop.py` :

   包含 `run_loop_steam` 函數，用來在 Steam 模式下執行情境。控制每一步的執行過程，並在每個步驟中進行遊戲邏輯的判斷，直至情境結束。
   
3. `C:\Users\yuhsu\anaconda3\envs\cmo\lib\site-packages\pycmo\lib\protocol.py` :

   負責處理與遊戲客戶端之間的協議，提供了處理視窗顯示、發送命令、檢查視窗是否存在等功能。例如，`window_exists` 函數檢查特定的彈出視窗是否出現，這在檢查情境是否結束時很有用。
   
4. `C:\Users\yuhsu\anaconda3\envs\cmo\lib\site-packages\pycmo\lib\tools.py` :

   包含一些工具函數，像是 `sleep` 等延遲控制，用來在操作之間進行暫停。這在等待遊戲進行或視窗顯示時非常重要。

### 其他重要
1. `C:\Users\yuhsu\pycmo-main\pycmo\configs\config.py` :

   建立並回傳一組設定參數（以字典形式），供 pycmo 框架在執行時使用。

   * 紀錄內容於 : `C:\Program Files (x86)\Steam\steamapps\common\Command - Modern Operations\ImportExport`
     * Steam demo.inst - 主要觀察資料文件
     * Steam demo_scen_has_ended.inst - 用於標記場景是否結束的文件

## 新增專案
`C:\Users\yuhsu\pycmo-main\scripts` :　為放置腳本的地方，於此建立要 run 的專案



