/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "cryptonight.h"
#include "cuda_device.hpp"
#include "Meraki_dag.h"
#include "CudaMeraki_gen.h"


void meraki_prepare(nvid_ctx *ctx, const void* cache, size_t cache_size, const void* dag_precalc, size_t dag_size, uint32_t height, const uint64_t* dag_sizes)
{
    constexpr size_t MEM_ALIGN = 1024 * 1024;

    if (cache_size != ctx->meraki_cache_size) {
        ctx->meraki_cache_size = cache_size;

        if (!dag_precalc) {
            if (cache_size > ctx->meraki_cache_capacity) {
                CUDA_CHECK(ctx->device_id, cudaFree(ctx->meraki_cache));

                ctx->meraki_cache_capacity = ((cache_size + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
                CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->meraki_cache, ctx->meraki_cache_capacity));
            }

            CUDA_CHECK(ctx->device_id, cudaMemcpy((uint8_t*)(ctx->meraki_cache), cache, cache_size, cudaMemcpyHostToDevice));
        }
    }

    if (dag_size != ctx->meraki_dag_size) {
        ctx->meraki_dag_size = dag_size;

        if (dag_size > ctx->meraki_dag_capacity) {
            CUDA_CHECK(ctx->device_id, cudaFree(ctx->meraki_dag));

            ctx->meraki_dag_capacity = ((dag_size + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
            CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->meraki_dag, ctx->meraki_dag_capacity));
        }

        if (dag_precalc) {
            CUDA_CHECK(ctx->device_id, cudaMemcpy((uint8_t*)(ctx->meraki_dag), cache, cache_size, cudaMemcpyHostToDevice));
        }

        constexpr int blocks = 8192;
        constexpr int threads = 32;

        const size_t cache_items = ((cache_size + 255) / 256) * 256 / sizeof(hash64_t);
        const size_t dag_items = dag_size / sizeof(hash64_t);

        uint4 light_words;
        light_words.w = ctx->meraki_cache_size / sizeof(hash64_t);
        Meraki_calculate_fast_mod_data(light_words.w, light_words.x, light_words.y, light_words.z);

        for (size_t i = dag_precalc ? cache_items : 0; i < dag_items; i += blocks * threads) {
            CUDA_CHECK_KERNEL(ctx->device_id, meraki_calculate_dag_item<<<blocks, threads>>>(
                i,
                (hash64_t*) ctx->meraki_dag,
                ctx->meraki_dag_size,
                (hash64_t*)(dag_precalc ? ctx->meraki_dag : ctx->meraki_cache),
                light_words
            ));
            CUDA_CHECK(ctx->device_id, cudaDeviceSynchronize());
        }

        if (dag_precalc) {
            CUDA_CHECK(ctx->device_id, cudaMemcpy((uint8_t*)(ctx->meraki_dag), dag_precalc, cache_items * sizeof(hash64_t), cudaMemcpyHostToDevice));
        }
    }

    constexpr uint32_t PERIOD_LENGTH = 3;
    const uint32_t period = height / PERIOD_LENGTH;

    if (ctx->meraki_period != period) {
        if (ctx->meraki_module) {
            cuModuleUnload(ctx->meraki_module);
        }

        std::vector<char> ptx;
        std::string lowered_name;
        Meraki_get_program(ptx, lowered_name, period, ctx->device_threads, ctx->device_arch[0], ctx->device_arch[1], dag_sizes);

        CU_CHECK(ctx->device_id, cuModuleLoadDataEx(&ctx->meraki_module, ptx.data(), 0, 0, 0));
        CU_CHECK(ctx->device_id, cuModuleGetFunction(&ctx->meraki_kernel, ctx->meraki_module, lowered_name.c_str()));

        ctx->meraki_period = period;

        Meraki_get_program(ptx, lowered_name, period + 1, ctx->device_threads, ctx->device_arch[0], ctx->device_arch[1], dag_sizes, true);
    }

    if (!ctx->meraki_stop_host) {
        CUDA_CHECK(ctx->device_id, cudaMallocHost(&ctx->meraki_stop_host, sizeof(uint32_t) * 2));
        CUDA_CHECK(ctx->device_id, cudaHostGetDevicePointer(&ctx->meraki_stop_device, ctx->meraki_stop_host, 0));
    }
}


void meraki_stop_hash(nvid_ctx *ctx)
{
    if (ctx->meraki_stop_host) {
        *ctx->meraki_stop_host = 1;
    }
}


namespace Meraki_Telestai {

void hash(nvid_ctx *ctx, uint8_t* job_blob, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t *skipped_hashes)
{
    dim3 grid(ctx->device_blocks);
    dim3 block(ctx->device_threads);

    uint32_t hack_false = 0;
    void* args[] = { &ctx->meraki_dag, &ctx->d_input, &target, &hack_false, &ctx->d_result_nonce, &ctx->meraki_stop_device };

    CUDA_CHECK(ctx->device_id, cudaMemcpy(ctx->d_input, job_blob, 40, cudaMemcpyHostToDevice));
    CUDA_CHECK(ctx->device_id, cudaMemset(ctx->d_result_nonce, 0, sizeof(uint32_t)));
    memset(ctx->meraki_stop_host, 0, sizeof(uint32_t) * 2);

    CU_CHECK(ctx->device_id, cuLaunchKernel(
        ctx->meraki_kernel,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, nullptr, args, 0
    ));
    CU_CHECK(ctx->device_id, cuCtxSynchronize());

    *skipped_hashes = ctx->meraki_stop_host[1];

    uint32_t results[16];
    CUDA_CHECK(ctx->device_id, cudaMemcpy(results, ctx->d_result_nonce, sizeof(results), cudaMemcpyDeviceToHost));

    if (results[0] > 15) {
        results[0] = 15;
    }

    *rescount = results[0];
    memcpy(resnonce, results + 1, results[0] * sizeof(uint32_t));
}

}
