# FeUdal架構總覽
FeUdal是一種階層式強化學習架構，主要由Manager和Worker兩個網絡組成：
## 1. 核心組件
1. Manager網絡：`Feudal_ManagerAgent`
   * 負責生成高層次的目標(goal)
   * 以較大的時間尺度運作
   * 從狀態輸入中提取特徵並生成目標向量
   * 使用`DilatedLSTMCell`處理時間序列資料
2. Worker網絡：`Feudal_WorkerAgent`
   * 接收Manager生成的目標
   * 負責選擇具體的動作來實現目標
   * 計算Q值來決定最佳動作
   * 同樣使用LSTM處理序列資料
3. Critic網絡：`FeUdalCritic`
   * 評估當前狀態的價值
   * 用於計算優勢(advantages)以優化Manager和Worker
## 2. 資料流程
在`sample_agent.py`的`action`方法中，可以看到完整流程：
1. 接收環境觀察資料
2. 提取狀態向量：`[B_lon, B_lat, B_heading, B_speed, A_lon, A_lat]`
3. Manager生成目標
4. Worker根據目標選擇動作
5. Critic評估狀態價值
6. 執行動作並記錄經驗
7. 定期訓練所有網絡
## 3. 訓練過程
在`train`方法中可以看到訓練邏輯：
1. Manager訓練：
   * 計算狀態變化與目標的相似度(cosine similarity)
   * 使用優勢(advantages)加權來優化目標生成
   * 鼓勵Manager生成與實際狀態變化一致的目標
2. Worker訓練：
   * 使用標準的TD學習更新Q值
   * 訓練Worker選擇能實現Manager目標的動作
3. Critic訓練：
   * 優化價值估計，最小化優勢平方誤差
4. 架構特點
   1. 分離關注點：
      * Manager關注長期目標和策略
      * Worker關注短期戰術決策
   2. 高效探索：
      * 通過分層結構改善探索效率
      * 由Manager引導的目標幫助Worker更有效地探索
   3. 時間抽象：
      * Manager可以設定較長期的目標
      * Worker執行一系列短期動作來實現這些目標
   4. 知識共享：
      * Manager的目標幫助Worker聚焦於有意義的行動
      * 形成有效的階層化決策過程
   5. 實現細節
      * 使用LSTM處理時序資料
      * 歸一化狀態資料以提高訓練穩定性
      * 計算cosine相似度來評估Worker是否遵循Manager的目標
      * 使用經驗回放來訓練網絡，保存完整episode
      * 設置epsilon-greedy探索策略，隨時間降低探索率
     

這種架構特別適合長期規劃和複雜決策任務，如在海戰模擬中控制艦船移動尋找目標。
