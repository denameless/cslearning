#include "asm_utils.h"
#include "interrupt.h"
#include "stdio.h"
#include "program.h"
#include "thread.h"
#include "sync.h"

STDIO stdio;
InterruptManager interruptManager;
ProgramManager programManager;

const int PHILOSOPHER_NUM = 5;
Semaphore chopsticks[PHILOSOPHER_NUM]; 

void philosopher(void *arg) {
    int id = (int)arg;
    int left = id % PHILOSOPHER_NUM;
    int right = (id + 1) % PHILOSOPHER_NUM;

    while (true) {
        // 思考
        printf("Philosopher %d is thinking...\n", id);
        int delay = 0xfffff;
        while (delay) --delay;

        // 获取第一根筷子
        if (id % 2 == 0) {
            chopsticks[left].P();
            printf("Philosopher %d took left chopstick %d\n", id, left);
            
            delay = 0xfffff;
            while (delay) --delay;
            
            chopsticks[right].P();
            printf("Philosopher %d took right chopstick %d\n", id, right);
        } else {
            chopsticks[right].P();
            printf("Philosopher %d took right chopstick %d\n", id, right);
            
            delay = 0xfffff;
            while (delay) --delay;
            
            chopsticks[left].P();
            printf("Philosopher %d took left chopstick %d\n", id, left);
        }

        // 就餐
        printf("Philosopher %d is eating noodles...\n", id);
        delay = 0xfffff;
        while (delay) --delay;

        // 释放筷子
        if (id % 2 == 0) {
            chopsticks[right].V();
            chopsticks[left].V();
        } else {
            chopsticks[left].V();
            chopsticks[right].V();
        }

        printf("Philosopher %d finished eating, released chopsticks\n", id);
    }
}

void first_thread(void *arg) {
    stdio.moveCursor(0);
    for (int i = 0; i < 25 * 80; ++i) stdio.print(' ');
    stdio.moveCursor(0);

    for (int i = 0; i < PHILOSOPHER_NUM; ++i) {
        chopsticks[i].initialize(1);
    }

    for (int i = 0; i < PHILOSOPHER_NUM - 1; ++i) {
        programManager.executeThread(philosopher, (void*)i, "Philosopher", 1);
    }
    programManager.executeThread(philosopher, (void*)6234, "Philo6234", 1);

    asm_halt();
}

extern "C" void setup_kernel() {
    interruptManager.initialize();
    interruptManager.enableTimeInterrupt();
    interruptManager.setTimeInterrupt((void *)asm_time_interrupt_handler);

    stdio.initialize();
    programManager.initialize();

    int pid = programManager.executeThread(first_thread, nullptr, "first thread", 1);
    if (pid == -1) {
        printf("cannot execute first thread\n");
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
