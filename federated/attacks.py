def sign_flip(update_dict, scale=5.0):
    # attacker sends inverted, amplified update
    return {k: -scale * v for k, v in update_dict.items()}
