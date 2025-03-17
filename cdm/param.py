import re

no_trained_para = [
    # 'input_blocks.0.0'
    # 'in_layers.2',
    # 'out_layers.3',
    # 'proj_in',
    # '.ff.net',
    # 'proj_out',
    # 'op',
    # 'skip_connection',
    # '.conv.',
    # 'out.0',
    # 'out.2'
    # 'transformer_blocks',

    'in_layers.0',
    # 'in_layers',
    'out_layers.0',
    # 'out_layers',
    'time_embed',
    'emb_layers',
    'norm',
    # 'attn1',
    # 'attn2',
    'norm1',
    'norm2',
    'norm3',
    'bias',
    '.out.0',

    # 'input_blocks.8.0',
    # 'input_blocks.9.0',
    # 'input_blocks.10.0',
    # 'input_blocks.11.0',
    # 'middle_block.0',
    # 'middle_block.2',
    # 'output_blocks.0.0',
    # 'output_blocks.1.0',
    # 'output_blocks.2.0',
    # 'output_blocks.3.0',
    # 'skip_connection',
    # 'conv'
]

# no_trained_para = [
# ]

control_trained_para = [
'input_hint_block.0',
'input_hint_block.1',
'input_hint_block.2',
'input_hint_block.3',
'input_hint_block.4',
'input_hint_block.5',
'input_hint_block.6',
'input_hint_block.7',
'input_hint_block.8',
'input_hint_block.9',
'input_hint_block.10',
'input_hint_block.11',
'input_hint_block.12',
'input_hint_block.13',
'input_hint_block.14',

'encoder.layers.0.self_attn',
'encoder.layers.0.self_attn.out_proj',
'encoder.layers.0.linear1',
'encoder.layers.0.linear2',

'encoder.layers.1.self_attn',
'encoder.layers.1.self_attn.out_proj',
'encoder.layers.1.linear1',
'encoder.layers.1.linear2',

'encoder.layers.2.self_attn',
'encoder.layers.2.self_attn.out_proj',
'encoder.layers.2.linear1',
'encoder.layers.2.linear2',

'encoder.layers.3.self_attn',
'encoder.layers.3.self_attn.out_proj',
'encoder.layers.3.linear1',
'encoder.layers.3.linear2',

'encoder.layers.4.self_attn',
'encoder.layers.4.self_attn.out_proj',
'encoder.layers.4.linear1',
'encoder.layers.4.linear2',

'encoder.layers.5.self_attn',
'encoder.layers.5.self_attn.out_proj',
'encoder.layers.5.linear1',
'encoder.layers.5.linear2',

'encoder.layers.6.self_attn',
'encoder.layers.6.self_attn.out_proj',
'encoder.layers.6.linear1',
'encoder.layers.6.linear2',

'encoder.layers.7.self_attn',
'encoder.layers.7.self_attn.out_proj',
'encoder.layers.7.linear1',
'encoder.layers.7.linear2'
]

def contains_any(target_str, substr_list):
    regex = '|'.join(map(re.escape, substr_list))
    return re.search(regex, target_str) is None

def sub_(name):
    return re.sub(r'\.[^.]*$', '', name)