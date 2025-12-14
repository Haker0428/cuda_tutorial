#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

void launch_flash_attention(float* output,
                            const float* q,
                            const float* k,
                            const float* v,
                            int batch_size,
                            int num_heads,
                            int seq_len,
                            int head_dim,
                            float scale);

#endif

