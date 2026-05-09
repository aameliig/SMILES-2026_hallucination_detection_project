import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Only last token from the last layer.
    Dimension = hidden_dim
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    real_mask = attention_mask.bool()
    if not real_mask.any():
        return torch.zeros(hidden_dim)

    last_idx = torch.where(real_mask)[0][-1]
    return hidden_states[-1, last_idx, :]

def extract_geometric_features(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Geometric features:
      - L2 norms per layer (n_layers)
      - cosine similarities between consecutive layers (n_layers-1)
      - normalized sequence length (1)
    Total: ~2*n_layers
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    real_mask = attention_mask.bool()
    if not real_mask.any():
        return torch.zeros(2 * n_layers - 1 + 1)

    last_idx = torch.where(real_mask)[0][-1]
    vecs = hidden_states[:, last_idx, :]

    # Norms (scaled)
    norms = torch.norm(vecs, p=2, dim=1) / (hidden_dim ** 0.5)   # (n_layers,)

    # Cosine similarities between consecutive layers
    cos_sims = []
    for l in range(n_layers - 1):
        cos = F.cosine_similarity(vecs[l].unsqueeze(0), vecs[l+1].unsqueeze(0))
        cos_sims.append(cos)
    cos_sims = torch.tensor(cos_sims) if cos_sims else torch.empty(0)

    # Normalized sequence length
    seq_len_norm = min(1.0, seq_len / 512.0)

    return torch.cat([norms, cos_sims, torch.tensor([seq_len_norm])])

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = True,
) -> torch.Tensor:
    token_feat = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geom = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([token_feat, geom], dim=0)
    else:
        return token_feat
