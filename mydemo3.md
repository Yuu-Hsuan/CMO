# how to open mydemo3

## 前置作業1
1. 打開cmo遊戲，點選:Create New Scenario
2. 點擊file，選擇load，選擇Browse
3. 開啟`"C:\Users\yuhsu\pycmo-main\scen\steam_demo.scen"`
4. 左下角`Enter scenario`
5. 選上方`Editor`，選`Lua Script Console`
6. 將`scripts\mydomo3\init`的程式的第6行改為:`local pycmo_path = 'C:/Users/yuhsu/pycmo-main/'`
7. 至`C:\Program Files (x86)\Steam\steamapps\common\Command - Modern Operations\ImportExport`將`Steam demo.inst`與`Steam demo_scen_has_ended.inst`改名(隨便改)
8. 將`scripts\mydomo3\init`的程式貼到下方方框中，點擊run，完成後儲存
9. 至`C:\Program Files (x86)\Steam\steamapps\common\Command - Modern Operations\ImportExport`將`Steam demo.inst`與`Steam demo_scen_has_ended.inst`改名為`MyScenario1.inst`與`MyScenario1_scen_has_ended.inst`
10. 將原先的`Steam demo.inst`與`Steam demo_scen_has_ended.inst`改回來

## 前置作業2
1. 加入`mydemo3`至`C:\Users\yuhsu\pycmo-main\scripts\`路徑
2. 加入學長的`lib`至`C:\Users\yuhsu\anaconda3\envs\cmo\Lib\site-packages\pycmo\`
3. 加入學長的`env`至`C:\Users\yuhsu\anaconda3\envs\cmo\Lib\site-packages\pycmo\`
4. 在`cmo`環境中輸入`pip install watchdog`
5. 完成 前置作業1

## step
1. 打開cmo
2. `conda activate cmo`
3. `cd "C:\Users\yuhsu\pycmo-main\scripts\mydemo3"`
4. 4. `python demo.py`
