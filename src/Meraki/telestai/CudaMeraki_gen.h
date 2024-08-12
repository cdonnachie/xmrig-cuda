#ifndef XMRIG_CUDAMERAKI_GEN_H
#define XMRIG_CUDAMERAKI_GEN_H

#include <cstdint>
#include <vector>
#include <string>

void Meraki_get_program(std::vector<char>& ptx, std::string& lowered_name, uint64_t period, uint32_t threads, int arch_major, int arch_minor, const uint64_t* dag_sizes, bool background = false);
void Meraki_calculate_fast_mod_data(uint32_t divisor, uint32_t& reciprocal, uint32_t& increment, uint32_t& shift);

#endif // XMRIG_CUDAMERAKI_GEN_H
