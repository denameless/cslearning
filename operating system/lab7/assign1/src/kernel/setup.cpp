#include "asm_utils.h"
#include "interrupt.h"
#include "stdio.h"
#include "program.h"
#include "thread.h"
#include "sync.h"
#include "memory.h"

// 屏幕IO处理器
STDIO stdio;
// 中断管理器
InterruptManager interruptManager;
// 程序管理器
ProgramManager programManager;
// 内存管理器
MemoryManager memoryManager;

void first_thread(void *arg)
{
    // 第1个线程不可以返回
    // stdio.moveCursor(0);
    // for (int i = 0; i < 25 * 80; ++i)
    // {
    //     stdio.print(' ');
    // }
    // stdio.moveCursor(0);
    int firstkerallocate = memoryManager.allocatePhysicalPages(AddressPoolType::KERNEL, 10);
    int firstuserallocate = memoryManager.allocatePhysicalPages(AddressPoolType::USER, 5);   
    int secondkerallocate = memoryManager.allocatePhysicalPages(AddressPoolType::KERNEL, 25);
    int seconduserallocate = memoryManager.allocatePhysicalPages(AddressPoolType::USER, 15);
    printf("first kernel allocate: %x, second kernel allocate: %x\n", firstkerallocate, secondkerallocate);
    printf("first user allocate: %x, second user allocate: %x\n", firstuserallocate, seconduserallocate);
    memoryManager.releasePhysicalPages(AddressPoolType::KERNEL, firstkerallocate, 10);
    memoryManager.releasePhysicalPages(AddressPoolType::USER, firstuserallocate, 5);
    int thirdkerallocate = memoryManager.allocatePhysicalPages(AddressPoolType::KERNEL, 5);
    int thirduserallocate = memoryManager.allocatePhysicalPages(AddressPoolType::USER, 20);
    printf("third kernel allocate: %x, third user allocate: %x", thirdkerallocate, thirduserallocate);
    asm_halt();
}

extern "C" void setup_kernel()
{

    // 中断管理器
    interruptManager.initialize();
    interruptManager.enableTimeInterrupt();
    interruptManager.setTimeInterrupt((void *)asm_time_interrupt_handler);

    // 输出管理器
    stdio.initialize();

    // 进程/线程管理器
    programManager.initialize();

    // 内存管理器
    memoryManager.initialize();

    // 创建第一个线程
    int pid = programManager.executeThread(first_thread, nullptr, "first thread", 1);
    if (pid == -1)
    {
        printf("can not execute thread\n");
        asm_halt();
    }

    ListItem *item = programManager.readyPrograms.front();
    PCB *firstThread = ListItem2PCB(item, tagInGeneralList);
    firstThread->status = RUNNING;
    programManager.readyPrograms.pop_front();
    programManager.running = firstThread;
    asm_switch_thread(0, firstThread);

    asm_halt();
}

