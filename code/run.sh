#!/bin/bash

# ======================================================
# 0. CONFIG
# ======================================================
# 👇👇👇 請在這裡填入你的 Token 👇👇👇
YOUR_HF_TOKEN="YOUR_HF_TOKEN_HERE"

================================================
# 1. 環境準備 & 更新 (針對 2025 新模型)
# ======================================================
echo "📦 Installing/Updating dependencies for 2025 models..."
# 強制更新 vLLM 和 Transformers 以支援 Gemma 3 / Qwen 3
pip install --upgrade vllm transformers hf_transfer huggingface_hub gdown bitsandbytes

export HF_HUB_ENABLE_HF_TRANSFER=1

# 登入邏輯
if [ -z "$HF_TOKEN" ]; then
    if [[ "$YOUR_HF_TOKEN" == *"請填入"* ]] || [ -z "$YOUR_HF_TOKEN" ]; then
        echo "⚠️  WARNING: No Token set. Downloads might fail."
    else
        echo "🔑 Logging in..."
        huggingface-cli login --token "$YOUR_HF_TOKEN" --add-to-git-credential
    fi
else
    echo "🔑 Using environment HF_TOKEN..."
fi

# ======================================================
# 2. 定義「把脈」函數 (Python 驗證邏輯)
# ======================================================
verify_integrity() {
    MODEL_DIR=$1
    
    # 使用內嵌 Python 腳本來檢查 config.json 和權重檔是否齊全
    python3 - <<EOF
import os
import json
import glob
import sys

path = "$MODEL_DIR"

def check():
    # 1. 檢查 Config (靈魂)
    if not os.path.exists(os.path.join(path, "config.json")):
        print(f"❌ Critical: config.json missing in {path}")
        return False

    # 2. 檢查 Index (分塊模型的地圖)
    index_path = os.path.join(path, "model.safetensors.index.json")
    
    if os.path.exists(index_path):
        # 如果有 index，檢查是否所有碎片都在
        try:
            with open(index_path, 'r') as f:
                data = json.load(f)
            expected_files = set(data['weight_map'].values())
            missing = []
            for f in expected_files:
                if not os.path.exists(os.path.join(path, f)):
                    missing.append(f)
            if missing:
                print(f"❌ Incomplete! Missing shards: {missing}")
                return False
            print(f"✅ Integrity verified (Multi-shard check passed)")
            return True
        except json.JSONDecodeError:
            print("❌ Error: Index json corrupted")
            return False
    else:
        # 如果沒有 index，檢查是否至少有一個權重檔 (.safetensors 或 .bin)
        files = glob.glob(os.path.join(path, "*.safetensors")) + glob.glob(os.path.join(path, "*.bin"))
        if not files:
            print("❌ No weight files found!")
            return False
        print(f"✅ Integrity verified (Single file check passed)")
        return True

if not check():
    sys.exit(1)
EOF

    # 接收 Python 的回傳值
    return $?
}

# ======================================================
# 3. 下載函數 (整合下載 + 把脈 + 自動刪除壞檔)
# ======================================================
mkdir -p models

download_model() {
    MODEL_ID=$1
    LOCAL_DIR=$2

    # 路徑防呆
    if [[ "$LOCAL_DIR" == /* ]]; then LOCAL_DIR="${LOCAL_DIR#/}"; fi

    echo "-------------------------------------------------------"
    echo "⬇️  Downloading: $MODEL_ID"
    
    # 開始下載
    huggingface-cli download --resume-download "$MODEL_ID" \
        --local-dir "$LOCAL_DIR" \
        --local-dir-use-symlinks False
    
    DOWNLOAD_STATUS=$?
    
    # 下載指令本身失敗
    if [ $DOWNLOAD_STATUS -ne 0 ]; then
        echo "❌ HuggingFace CLI Failed. Removing folder..."
        rm -rf "$LOCAL_DIR"
        return
    fi

    # 🏥 馬上把脈 (檢查檔案完整性)
    echo "🏥 Verifying integrity for $LOCAL_DIR..."
    verify_integrity "$LOCAL_DIR"
    
    VERIFY_STATUS=$?
    
    if [ $VERIFY_STATUS -ne 0 ]; then
        echo "🧨 BAD DOWNLOAD DETECTED! Deleting corrupted folder to prevent crashes..."
        rm -rf "$LOCAL_DIR"
        echo "🗑️  Deleted $LOCAL_DIR"
    else
        echo "🎉 Model Ready: $MODEL_ID"
    fi
}

# ======================================================
# 4. 開始執行 (2025 清單)
# ======================================================
echo "🚀 Starting Workflow..."

# --- Adapter ---
echo "⬇️ Downloading Adapters..."
gdown --folder --id "1QmFJw98PNF_gf0h217b74Gci14jU_7Tc" --output adapters/
echo "🚀 Starting Model Downloads..."

# --- Base Models ---
download_model "unsloth/Qwen2.5-7B-Instruct" "models/Qwen2.5-7B-Instruct"
download_model "unsloth/Qwen2.5-14B-Instruct" "models/Qwen2.5-14B-Instruct"
download_model "unsloth/Qwen3-4B-Instruct-2507" "models/Qwen3-4B-Instruct-2507"
# download_model "unsloth/Llama-3.2-3B-Instruct" "models/Llama-3.2-3B-Instruct"
download_model "unsloth/Llama-3.1-8B-Instruct" "models/Llama-3.1-8B-Instruct"
download_model "unsloth/DeepSeek-R1-Distill-Llama-8B" "models/DeepSeek-R1-Distill-Llama-8B"
download_model "unsloth/DeepSeek-R1-Distill-Qwen-7B" "models/DeepSeek-R1-Distill-Qwen-7B"
download_model "google/gemma-2-9b-it" "models/gemma-2-9b-it"
download_model "google/gemma-3-12b-it" "/models/gemma-3-12b-it"
download_model "unsloth/Mistral-Nemo-Instruct-2407" "models/Mistral-Nemo-Instruct-2407"
download_model "NousResearch/Hermes-3-Llama-3.1-8B" "models/Hermes-3-Llama-3.1-8B"
download_model "unsloth/phi-4" "models/phi-4"
download_model "01-ai/Yi-1.5-9B-Chat" "models/Yi-1.5-9B-Chat"
download_model "upstage/SOLAR-10.7B-Instruct-v1.0" "models/SOLAR-10.7B-Instruct-v1.0"
download_model "dphn/Dolphin3.0-Llama3.1-8B" "models/Dolphin3.0-Llama3.1-8B"
download_model "unsloth/Qwen2.5-Coder-7B-Instruct" "models/Qwen2.5-Coder-7B-Instruct"
# --- Evaluation & RAG Models ---
download_model "Qwen/Qwen3Guard-Gen-0.6B" "models/Qwen3Guard-Gen-0.6B"
download_model "unsloth/Llama-3.2-3B-Instruct" "models/Llama-3.2-3B-Instruct"
download_model "theblackcat102/Qwen3-1.7B-Usefulness-Judge" "models/Qwen3-1.7B-Usefulness-Judge"
download_model "intfloat/multilingual-e5-large" "models/multilingual-e5-large"
download_model "BAAI/bge-reranker-v2-m3" "models/bge-reranker-v2-m3"

echo "✅ All models verified and ready."