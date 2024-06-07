from llama_cpp import Llama


def load_model(model_path):
    model = Llama(
        model_path,
        n_gpu_layers=-1,
        verbose=False,
        use_mlock=True,
        logits_all=True,
        n_ctx=0,
    )
    return model
