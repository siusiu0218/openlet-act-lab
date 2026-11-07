"""Shared policy configuration entry point for preparation and training."""
def load_act_config(path, device):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    # Serialized config includes the registry discriminator `type: act`.
    # Load via the registry base, as PreTrainedPolicy.from_pretrained does.
    config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    if not isinstance(config, ACTConfig):
        raise TypeError('Expected an ACT configuration')
    config.device = device
    config.push_to_hub = False
    return config
