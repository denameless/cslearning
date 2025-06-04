#include "os_type.h"

template<typename T>
void swap(T &x, T &y) {
    T z = x;
    x = y;
    y = z;
}


void itos(char *numStr, uint32 num, uint32 mod) {
    // 只能转换2~26进制的整数
    if (mod < 2 || mod > 26 || num < 0) {
        return;
    }

    uint32 length, temp;

    // 进制转换
    length = 0;
    while(num) {
        temp = num % mod;
        num /= mod;
        numStr[length] = temp > 9 ? temp - 10 + 'A' : temp + '0';
        ++length;
    }

    // 特别处理num=0的情况
    if(!length) {
        numStr[0] = '0';
        ++length;
    }

    // 将字符串倒转，使得numStr[0]保存的是num的高位数字
    for(int i = 0, j = length - 1; i < j; ++i, --j) {
        swap(numStr[i], numStr[j]);
    }
    

    numStr[length] = '\0';
}

void ftos(char *buf, double num, int precision) {
    int idx = 0;
    
    // 处理负数
    if (num < 0) {
        buf[idx++] = '-';
        num = -num;
    }

    // 提取整数部分
    int integer = (int)num;
    double fraction = num - integer;

    // 转换整数部分
    char int_buf[32];
    itos(int_buf, integer, 10);
    for (int i = 0; int_buf[i]; ++i) {
        buf[idx++] = int_buf[i];
    }

    // 添加小数点
    buf[idx++] = '.';

    // 转换小数部分（逐位乘以10取整）
    for (int i = 0; i < precision; ++i) {
        fraction *= 10;
        int digit = (int)fraction;
        buf[idx++] = '0' + digit;
        fraction -= digit;
    }

    buf[idx] = '\0'; // 终止字符串
}
