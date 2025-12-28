---
title: README

---

# ADL 2025 Final Project: Jailbreak Olympics

本專案目標是在解決 LLM Jailbreak 任務中的多樣性與成功率問題。我們提出了一種 **Hybrid Attack Architecture (混合式攻擊架構)**，整合了 **RAG (Retrieval-Augmented Generation)** 檢索增強、**Multi-Model Arena (多模型競技場)** 以及 **Local Sequential Evaluator (本地序列評測)**，透過自動化改寫惡意提示詞 (Prompt Rewriting) 來繞過現代 LLM 的防禦機制。

---

##  檔案結構 (File Structure)

| 檔案名稱 | 說明 |
| :--- | :--- |
| code/run.sh | **環境建置與下載腳本**。負責安裝 LLM, Unsloth 等相依套件，驗證模型完整性 (Integrity Check)，並從 Hugging Face 與 Google Drive 自動下載所有 Base Models 與 LoRA Adapters。 |
| code/algorithms.py | **核心推論 (Inference) 程式**。實作完整的攻擊 Pipeline：RAG Retrieval $\to$ Hybrid Generation $\to$ Sequential Evaluation $\to$ Selection Strategy。 |
| code/lora_rewritter_training.ipynb | **模型訓練筆記**。展示如何使用 Unsloth 框架對 Qwen/Llama 等模型進行 LoRA 微調，以學習 "Creative Screenwriter" 的攻擊風格。 |
| results/ | **實驗結果**。包含不同成本設定下的重寫提示詞結果 (.jsonl 格式)。 |

---

## 1. 環境建置與模型準備 (Environment Setup)

為了確保實驗結果的可再現性 (Reproducibility)，我們提供了一鍵式腳本。

### 前置作業
請確保你擁有一個有效的 Hugging Face Token (需有 Read 權限) 以驗證 Gated Models (如 Llama-3, Mistral 等)。

本專案依賴於 [2025-ADL-Final-Challenge-Release](https://github.com/yenshan0530/2025-ADL-Final-Challenge-Release) 提供的基礎架構。

請先 clone 該儲存庫以取得必要的執行腳本 (
un_inference.py, 
un_eval.py) 與資料集：
`ash
git clone https://github.com/yenshan0530/2025-ADL-Final-Challenge-Release.git
cd 2025-ADL-Final-Challenge-Release
`

下載完成後，請將本專案的 code/run.sh 複製到 src 資料夾底下。

### 執行安裝
`ash
# 1. 賦予腳本執行權限
chmod +x src/run.sh

# 2. 執行腳本 (請確保硬碟空間 > 100GB 以存放權重)
# 腳本內建自動修復機制，若下載中斷會自動重試
bash src/run.sh
`

Note: 腳本會安裝 vllm, unsloth, transformers, bitsandbytes 等高效能推論與訓練套件，並啟用 HF_HUB_ENABLE_HF_TRANSFER 加速下載。  

## 2. 使用方法

請將本專案的 code/algorithms.py 取代原本的 src/algorithms.py，並執行：

`ash
# 使用簡單版評估 (Single Model + Rerank)
python run_inference.py --algorithm evaluate_rewrite_simple
`

`ash
# 使用完整版評估 (Full Pipeline)
python run_inference.py --algorithm evaluate_rewrite
`

## 3. 系統架構與推論流程 (Inference Pipeline)

lgorithms.py 實作了本專案的核心邏輯 evaluate_rewrite(toxic_prompt)，流程分為四個階段：

### Phase 1: Context-Aware RAG Retrieval
利用 Few-shot Learning 增強攻擊的隱蔽性。
* **Retriever:** intfloat/multilingual-e5-large
* **Reranker:** BAAI/bge-reranker-v2-m3
* **流程:** 針對輸入的惡意 Prompt，從知識庫 (RAG_KNOWLEDGE_BASE) 檢索語意最接近的 3 個「成功越獄範例」，並進行重排序 (Re-ranking) 以確保品質。

### Phase 2: Hybrid Arena V4 (Generation)
採用多模型並行生成的策略 (Ensemble Strategy) 以最大化攻擊多樣性。我們部署了以下模型陣容：
* **Flagship (高智商/強邏輯):** Qwen-2.5-14B, Phi-4-14B, Mistral-Nemo-12B
* **Specialist (特定領域/無審查):** Hermes-3, Yi-1.5, Solar-10.7B
* **Base/Distilled:** Llama-3.1, Gemma-2, DeepSeek-R1-Distill
* **技術細節:** 針對不同模型家族 (如 Gemma, Mistral) 動態切換 Chat Template，並支援載入特定的 LoRA Adapters 進行風格遷移。

### Phase 3: Local Sequential Evaluator (Simulation)
在提交前進行本地端模擬，過濾無效攻擊以節省 Query Budget。
1.  **Safety Guard:** 使用 Qwen/Qwen3Guard-Gen-0.6B 攔截顯著的惡意請求 (Safety Pattern Matching)。
2.  **Chat Simulation:** 使用 Llama-3.2-3B 作為 Target Model Proxy 模擬對話回應。
3.  **Relevance Judge:** 使用 Qwen3-1.7B-Usefulness-Judge (CoT-based Judge) 進行二元分類 (YES/NO)，判斷回應是否切題。

### Phase 4: Tier-based Selection Strategy
當多個候選攻擊通過檢測時，採用以下優先級進行最終決策：
1.  **Score Priority:** 優先選擇 Safety Label = Safe 且 Relevance Score = 1.0 的結果。
2.  **Model Tier:** 優先選擇來自 **Tier 1 模型** (如 Qwen-14B) 的產出，因大模型邏輯較嚴謹，較易繞過官方的 Private Evaluation。
3.  **Length Heuristic:** 在同分情況下，選擇 **長度較長** 的 Prompt (通常包含更多 Context Padding 與劇情鋪陳，能稀釋惡意意圖)。

---

## 4. 模型訓練細節 (Training Details)

若需重現我們的 LoRA Adapters (如 dapters/qwen_2.5_7b)，請參考 code/lora_rewritter_training.ipynb。

### Training Configuration
我們使用 **Unsloth** 框架進行高效微調：
* **Quantization:** 4-bit (QLoRA)
* **LoRA Rank (r):** 16
* **LoRA Alpha:** 32
* **Target Modules:** ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

### Hyperparameters
* **Learning Rate:** 2e-4
* **Batch Size:** 4 (with Gradient Accumulation)
* **Max Steps:** 100~150 (針對 Style Transfer 任務，少量步數即可收斂，避免過擬合)

### Data Strategy
* **Dataset:** ADL Public Dataset (25W candidates)
* **Prompt Engineering:** 訓練時將 Jailbreak System Prompt (e.g., *"You are a creative screenwriter..."*) 強制注入，訓練模型將任何惡意指令轉化為虛構劇本。

---

## 硬體需求與注意事項 (Requirements)

* **GPU Memory:** 推論階段需同時載入多個 14B 模型，建議使用 **24GB VRAM** 以上的 GPU (如 RTX 3090/4090, A10g, L4)。
* **Context Length:** 為避免 OOM，Inference 過程中已對 Context Length 進行優化與限制。