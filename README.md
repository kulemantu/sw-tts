# Swahili TTS Fine-tuning for Cloud GPU

This repository contains the necessary code and configuration to fine-tune the `facebook/mms-tts-swh` model using the Mozilla Common Voice Swahili dataset on a cloud GPU environment (e.g., Google Colab, AWS SageMaker, GCP AI Platform, Kaggle Kernel).

It is adapted from the [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) repository.

## Prerequisites

*   Cloud environment with a CUDA-enabled GPU (e.g., NVIDIA T4, V100, A100).
*   Python 3.8+
*   Git
*   Hugging Face account and authentication token ([Settings -> Access Tokens](https://huggingface.co/settings/tokens)).
*   (Optional) Weights & Biases account for experiment tracking ([wandb.ai](https://wandb.ai/)).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd swahili-tts-cloud-gpu
    ```

2.  **Create a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    # Verify torch is installed with CUDA support
    # python -c "import torch; print(torch.cuda.is_available())"
    ```

4.  **Build Monotonic Alignment Search:**
    This requires C build tools (e.g., `build-essential` on Debian/Ubuntu).
    ```bash
    cd monotonic_align
    python setup.py build_ext --inplace
    cd ..
    ```

5.  **Log in to Hugging Face:**
    ```bash
    huggingface-cli login
    # Enter your token with write permissions
    ```

6.  **(Optional) Log in to Weights & Biases:**
    ```bash
    wandb login
    # Enter your API key
    ```

## Fine-tuning Process

1.  **Prepare the Base Model:**
    Convert the original MMS checkpoint to include the discriminator. This saves the required model files into the `./models/base_model` directory.
    ```bash
    python convert_original_discriminator_checkpoint.py --language_code sw --pytorch_dump_folder_path ./models/base_model
    ```
    *(Note: The language code here is `sw`, consistent with the dataset config name).* 

2.  **Accept Dataset Terms:**
    You *must* have accepted the terms for the `mozilla-foundation/common_voice_17_0` dataset on the Hugging Face website while logged in. The script will fail otherwise during data download.
    Visit [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) and accept the terms.

3.  **Review Configuration:**
    Open `training_config.json` and adjust parameters if needed:
    *   `filter_on_speaker_id`: Change `"female"` to `"male"` or `"other"` if desired.
    *   `hub_model_id`: **Crucially, change `"your-username/swahili-mms-female-voice-cloud"`** to your actual Hugging Face username and desired model name.
    *   `per_device_train_batch_size`: Adjust based on your GPU's VRAM (e.g., lower for T4, higher for A100).
    *   `gradient_accumulation_steps`: Adjust inversely to batch size.
    *   `fp16`: Set to `false` if your GPU doesn't support FP16 well.
    *   `push_to_hub`: Set to `false` if you don't want to automatically upload the final model.

4.  **Configure Accelerate:**
    Set up `accelerate` for your GPU environment.
    ```bash
    accelerate config
    ```
    *   Choose `This machine`.
    *   Choose `multi-GPU` if you have multiple GPUs, otherwise `No distributed training`.
    *   Follow prompts, generally accepting defaults (saying `No` to DeepSpeed, Dynamo, etc.).
    *   Select `fp16` if you enabled it in the config and your GPU supports it.

5.  **Run Fine-tuning:**
    Launch the training script using `accelerate`.
    ```bash
    accelerate launch run_vits_finetuning.py training_config.json
    ```

## Monitoring

*   **Terminal Output:** Track progress directly in the terminal.
*   **TensorBoard:** Launch TensorBoard to view loss curves and other metrics.
    ```bash
    # Run in a separate terminal or tmux/screen session
    tensorboard --logdir models/finetuned_model/runs
    ```
*   **Weights & Biases:** If logged in, monitor experiments on the W&B dashboard.

## Using the Fine-tuned Model

Once training is complete (or you want to test a checkpoint):

1.  The best model is saved in `./models/finetuned_model`.
2.  If `push_to_hub` was true, it's also uploaded to your Hugging Face Hub repository.

Use the model with the `transformers` pipeline:

```python
from transformers import pipeline
import scipy.io.wavfile
import torch

# Use the Hub ID or the local path to your final model/checkpoint
model_id = "your-username/swahili-mms-female-voice-cloud" # Or "./models/finetuned_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

synthesiser = pipeline("text-to-speech", model=model_id, device=device)

text = "Habari gani? Hii ni mfano wa sauti iliyotengenezwa kutoka kwenye mfano ulioboreshwa."
output_file = "finetuned_cloud_output.wav"

# Generate speech
speech = synthesiser(text)

# Save audio
scipy.io.wavfile.write(
    output_file, 
    rate=speech["sampling_rate"], 
    data=speech["audio"]
)

print(f"Audio saved to {output_file}")
```

## Directory Structure

```
swahili-tts-cloud-gpu/
├── .git/
├── .gitignore
├── README.md
├── requirements.txt
├── training_config.json
├── run_vits_finetuning.py
├── convert_original_discriminator_checkpoint.py
├── monotonic_align/           # Code for alignment
│   └── ...
├── utils/                     # Utility functions
│   └── ...
├── data/                      # For downloaded/processed data (ignored by git)
│   └── .gitkeep
└── models/                    # For model files (ignored by git)
    ├── base_model/            # Converted base model for training
    │   └── .gitkeep
    └── finetuned_model/       # Output directory for checkpoints & final model
        └── .gitkeep
``` 