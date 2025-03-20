#pragma once
#include <windows.h>
#include <combaseapi.h>
#include "sandbox.h"

// COM API 模拟函数声明
void Api_CoInitializeEx(void* sandbox, uc_engine* uc, uint64_t address);
void Api_CoCreateInstance(void* sandbox, uc_engine* uc, uint64_t address);
void Api_VariantInit(void* sandbox, uc_engine* uc, uint64_t address);
void Api_VariantClear(void* sandbox, uc_engine* uc, uint64_t address);
void Api_SysAllocString(void* sandbox, uc_engine* uc, uint64_t address);