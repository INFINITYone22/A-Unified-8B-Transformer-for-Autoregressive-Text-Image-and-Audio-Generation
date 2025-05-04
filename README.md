# Unified Multi-Modal Transformer (Text, Image, Audio) - PyTorch Implementation

This repository contains a PyTorch implementation of a large-scale, unified, decoder-only Transformer model designed for autoregressive generation across multiple modalities: text, images (via discrete VQ-tokens), and audio (via discrete VQ-tokens). The architecture is inspired by recent advancements in large multi-modal models and is designed with scalability towards ~8 billion parameters in mind.

## Architecture Deep Dive

This implementation focuses on a robust and efficient Transformer backbone capable of handling interleaved sequences from different modalities. Key architectural components include:

1.  **Core Backbone:** A **decoder-only Transformer** architecture forms the foundation, processing input sequences autoregressively to predict the next token, regardless of its modality.
2.  **Unified Modality Handling:** The model processes sequences containing a mix of:
    * **Text Tokens:** Standard subword units (e.g., BPE, SentencePiece).
    * **Image Tokens:** Discrete indices from a pre-trained Vector Quantized (VQ) image tokenizer (e.g., VQ-VAE, VQGAN).
    * **Audio Tokens:** Discrete indices from a pre-trained neural audio codec (e.g., SoundStream, EnCodec).
    The model utilizes separate embedding layers for each modality's vocabulary, augmented by **modality embeddings** to provide context about the type of token being processed at each position.
3.  **Efficient Attention:** Employs multi-head **Causal Self-Attention**.
    * Leverages **FlashAttention** (v2 recommended) if available, significantly reducing memory footprint (from quadratic to near-linear scaling) and accelerating computation, crucial for handling long sequences common in image and audio data.
    * Includes a fallback to PyTorch's native `scaled_dot_product_attention` (SDPA) for broader compatibility (efficient in PyTorch 2.0+).
4.  **Advanced Positional Encoding:** Implements **Rotary Positional Embeddings (RoPE)**. RoPE is applied directly to query and key vectors within the attention mechanism, offering improved generalization across different sequence lengths and enhanced model stability compared to traditional absolute or learned embeddings.
5.  **Normalization:** Utilizes **Pre-Layer Normalization** (Norm before Attention/FFN) for improved gradient flow and training stability, a common practice in modern large Transformer models.
6.  **Output Projection:** Features **separate linear output heads** for predicting image VQ tokens and audio VQ tokens, directly projecting the final hidden states to the respective vocabularies. *Note: Text token prediction requires separate handling, potentially via weight tying with the text embedding layer or a dedicated text head.*
7.  **External Tokenization:** This implementation focuses on the core Transformer. It assumes that raw text, images, and audio have been converted into sequences of discrete token IDs by **external, pre-trained tokenizers**. The model consumes these IDs and corresponding modality identifiers.
8.  **Scalability:** The provided configuration (`hidden_size=4096`, `num_layers=36`, `num_attention_heads=32`, `ffn_hidden_size=16384`) aligns with models in the **~8 Billion parameter scale**. The architectural choices (FlashAttention, RoPE, standard block design) ensure the model is scalable, and the code is compatible with techniques like mixed-precision training (demonstrated in the example usage).

## Code Structure

The implementation is organized into modular components:

* `UnifiedTransformerConfig`: Dataclass for managing all hyperparameters.
* `RotaryPositionalEmbeddings`: Standalone RoPE implementation.
* `CausalSelfAttention`: Attention module integrating RoPE and FlashAttention/SDPA.
* `FeedForward`: Standard FFN block.
* `TransformerBlock`: Combines Attention, FFN, LayerNorm, and residual connections.
* `UnifiedTransformer`: The main model class orchestrating embeddings, Transformer layers, and output heads.

## Key Features

* Unified sequence processing for text, image (VQ), and audio (VQ) tokens.
* Efficient and scalable decoder-only Transformer architecture.
* Optimized Causal Self-Attention using FlashAttention (recommended) or PyTorch SDPA.
* Rotary Positional Embeddings (RoPE) for robust sequence modeling.
* Modality embeddings to differentiate token types.
* Separate output heads tailored for image and audio discrete token generation.
* Configurable hyperparameters targeting ~8B parameter scale.
* Standard practices like Pre-Layer Normalization and careful weight initialization.
* Includes example usage demonstrating instantiation and forward pass.

## Setup & Usage

1.  **Dependencies:**
    * `torch >= 2.0` (for SDPA fallback and general features)
    * `flash-attn >= 2.0` (Recommended for performance, install via `pip install flash-attn --no-build-isolation`)
    * `transformers` or `sentencepiece` (For text tokenization - *external*)
    * Relevant libraries for your chosen Image VQ and Audio Codec models (*external*)

2.  **Instantiation:**
    ```python
    from model import UnifiedTransformer, UnifiedTransformerConfig

    # Configure based on your tokenizers and desired scale
    config = UnifiedTransformerConfig(
        text_vocab_size=50257,
        image_vocab_size=16384,
        audio_vocab_size=1024,
        # Adjust dimensions (hidden_size, num_layers, etc.) as needed
        # ... other hyperparameters ...
        use_flash_attention=True, # Set based on availability
        use_rope=True
    )

    model = UnifiedTransformer(config).to(config.device)
    ```

3.  **Forward Pass:**
    ```python
    # Assume token_ids and modality_ids are prepared torch tensors
    # token_ids shape: (batch_size, sequence_length)
    # modality_ids shape: (batch_size, sequence_length) [0=text, 1=image, 2=audio]

    image_logits, audio_logits = model(token_ids, modality_ids)

    # image_logits shape: (batch_size, sequence_length, image_vocab_size)
    # audio_logits shape: (batch_size, sequence_length, audio_vocab_size)
    ```
    *(Refer to the `if __name__ == "__main__":` block in the code for a runnable example.)*

## Future Work / Roadmap

* Integration with specific text (e.g., SentencePiece), image (e.g., VQGAN), and audio (e.g., EnCodec) tokenizers.
* Development of a comprehensive training loop, including loss calculation strategies for the multi-head setup.
* Implementation of gradient checkpointing to further reduce memory usage during training of large models.
* Adding sophisticated sampling methods (top-k, top-p, temperature) for high-quality generation.
* Establishing evaluation metrics and benchmarking procedures.

## License

**Copyright (c) 2025 INFINITYone22**

**All Rights Reserved.**

Unauthorized copying, modification, distribution, or use of this code, or any part thereof, via any medium, is strictly prohibited unless explicit permission is granted by the copyright holder. This software is provided "as is" without warranty of any kind, express or implied.

---

## License Explanation for You

To retain complete ownership and control over your code and prevent others from using, modifying, or distributing it without your permission, the best approach on GitHub is **not to include a standard open-source license file** (like MIT, Apache 2.0, or GPL).

Here's why and what it means:

1.  **Default Copyright:** When you create original work (like code), you automatically hold the copyright under international law (e.g., the Berne Convention). This copyright grants you exclusive rights to reproduce, distribute, modify, and perform the work.
2.  **No License = All Rights Reserved:** By *not* providing an open-source license, you are not granting any of those exclusive rights to others. The code is effectively proprietary.
3.  **GitHub's Terms of Service:** GitHub's ToS allow users to view and fork public repositories *on the GitHub platform*. This is a limited right necessary for GitHub's function; it does **not** grant permission to use, modify, or distribute the code outside of that specific GitHub context (e.g., in another project, commercially, etc.).
4.  **README Statement:** The "License" section I included in the README above clearly states "All Rights Reserved" and explicitly prohibits unauthorized actions. This reinforces your intention and clarifies the terms for anyone viewing the repository.

**Implications:**

* **Maximum Control:** You retain full legal control over the code.
* **Limited Collaboration:** Others cannot legally contribute back (unless you make a separate agreement) or build upon your work.
* **Limited Use:** Others cannot legally use your code in their projects (open source or commercial).
* **Recruiter Perception:** While it shows you can write complex code, recruiters specifically looking for open-source contributions or experience with collaborative development might see this differently than a project with a standard OS license. However, the detailed technical description in the README should still be impressive.

This "All Rights Reserved" approach directly addresses your requirement to retain complete ownership. Just be aware of the trade-offs regarding collaboration and open-source participation.
