### 1.Policy Gradient 方法：

* Feudal_WorkerAgent 使用來自 worker RNN（LSTMCell）的 policy logits 和動作嵌入來計算動作的機率（使用 Categorical 分佈），這是典型的 policy gradient 方法export_observationinit。

* 動作是通過從 logits 中抽樣來選擇的，這也顯示出它使用 policy gradient 來進行動作選擇init。

### 2.Actor-Critic：

* 系統中使用了兩個不同的 critic：worker_critic 和 manager_critic，它們負責估算 狀態的價值（critic 角色），並且指導 actor（worker 和 manager）的訓練init。

* critic loss（如 value_loss）被分別計算給 worker 和 manager，這顯示出典型的 actor-critic 結構，其中 actor 學習如何最大化回報，而 critic 則評估其所選擇的動作init。

### 3.Feudal 架構：

* 系統使用了層級結構，包括 manager 和 worker，每個角色都有自己的學習網絡。manager 生成高層次的目標，worker 根據這些目標來執行動作。這是一種層級強化學習方法，其中 manager 監控並指導 worker 的學習過程export_observationinit。

### 4.On-policy 學習：

* 該代理使用了 on-policy learning 方法，並且運用了 Generalized Advantage Estimation (GAE) 來計算 worker 和 manager 的優勢並更新他們的策略init。
* 

總結來說，這個系統確實結合了 policy gradient 和 actor-critic 方法，並使用了層級強化學習架構，實現了 manager 和 worker 之間的協同學習。
