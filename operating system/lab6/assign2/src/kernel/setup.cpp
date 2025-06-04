#include "asm_utils.h"
#include "interrupt.h"
#include "stdio.h"
#include "program.h"
#include "thread.h"
#include "sync.h"

// Screen IO handler
STDIO stdio;
// Interrupt manager
InterruptManager interruptManager;
// Program manager
ProgramManager programManager;

int plate_capacity = 5;
int current_cakes = 0;
int matcha_cakes = 0;  // 抹茶蛋糕数量
int mango_cakes = 0;   // 芒果蛋糕数量
bool error = false;

Semaphore plate_sem;  // 用于盘子操作的信号量
Semaphore matcha_sem; // 用于抹茶蛋糕的信号量
Semaphore mango_sem;

void service_a(void *arg) {
    int i = 100;
    while (i--) {
        if(error) break;

        plate_sem.P();
        if (matcha_cakes == 0 && current_cakes < plate_capacity) {
            int delay = 0xffffff;
            while (delay) --delay;

            matcha_cakes++;
            current_cakes++;
            printf("Service A6234: Added 1 matcha. Total: %d (Matcha: %d, Mango: %d)\n", 
                  current_cakes, matcha_cakes, mango_cakes);
	    matcha_sem.V();
        }
        plate_sem.V();
        int newdelay = 0xffffff;
        while (newdelay) --newdelay;
    }
}

void service_b(void *arg) {
    int i = 100;
    while (i--) {
        if(error) break;
        plate_sem.P();

        if (mango_cakes == 0 && current_cakes < plate_capacity) {
            int delay = 0xffffff;
            while (delay) --delay;

            mango_cakes++;
            current_cakes++;
            printf("Service B: Added 1 mango. Total: %d (Matcha: %d, Mango: %d)\n", 
                  current_cakes, matcha_cakes, mango_cakes);
	    mango_sem.V();
        }
        plate_sem.V();
        int newdelay = 0xffffff;
        while (newdelay) --newdelay;
    }
}

void male_guest(void *arg) {
    int id = (int)arg;
    int i = 10;
    while (i--) {
        if(error) break;
        matcha_sem.P(); // 等待抹茶蛋糕可用
        plate_sem.P();

        if (matcha_cakes > 0) {
            int delay = 0xfffffff;
            while (delay) --delay;
            
            matcha_cakes--;
            current_cakes--;
            printf("Male %d: Took 1 matcha. Remaining: %d (Matcha: %d, Mango: %d)\n", 
                  id, current_cakes, matcha_cakes, mango_cakes);
        } 
        else if(matcha_cakes < 0) {
            printf("error: matcha_cakes = %d\n", matcha_cakes);
            error = true;
        }
        else if (current_cakes < plate_capacity) {
            printf("Male %d: Requesting matcha ", id);
            int delay = 0xffffff;
            while (delay) --delay;
        }
	plate_sem.V();
    }
}

void female_guest(void *arg) {
    int id = (int)arg;
    int i = 10;
    while (i--) {
        if(error) break;
        mango_sem.P(); // 等待芒果蛋糕可用
        plate_sem.P();
 
        if (mango_cakes > 0) {
            int delay = 0xfffffff;
            while (delay) --delay;
            
            mango_cakes--;
            current_cakes--;
            printf("Female %d: Took 1 mango. Remaining: %d (Matcha: %d, Mango: %d)\n", 
                  id, current_cakes, matcha_cakes, mango_cakes);
        } 
        else if(mango_cakes < 0) {
            printf("error: mango_cakes = %d\n", mango_cakes);
            error = true;
        }
        else if (current_cakes < plate_capacity) {
            printf("Female %d: Requesting mango ", id);
            int delay = 0xffffff;
            while (delay) --delay;
        }
	plate_sem.V();
    }
}

void first_thread(void *arg) {
    // 清屏
    stdio.moveCursor(0);
    for (int i = 0; i < 25 * 80; ++i) {
        stdio.print(' ');
    }
    stdio.moveCursor(0);

    current_cakes = 0;
    matcha_cakes = 0;
    mango_cakes = 0;
    plate_sem.initialize(1);   // 互斥信号量，初始为1
    matcha_sem.initialize(0);  // 初始没有抹茶蛋糕
    mango_sem.initialize(0);   // 初始没有芒果蛋糕
    programManager.executeThread(service_a, nullptr, "Service A", 1);
    programManager.executeThread(service_b, nullptr, "Service B", 1);

    for (int i = 1; i <= 7; i+=2) {
        programManager.executeThread(male_guest, (void*)i, "Male Guest", 1);
        programManager.executeThread(female_guest, (void*)i + 1, "Female Guest", 1);
    }

    for (int i = 9; i <= 10; ++i) {
        programManager.executeThread(female_guest, (void*)i, "Female Guest", 1);
    }

    asm_halt();
}

extern "C" void setup_kernel()
{
    // Initialize interrupt manager
    interruptManager.initialize();
    interruptManager.enableTimeInterrupt();
    interruptManager.setTimeInterrupt((void *)asm_time_interrupt_handler);

    // Initialize IO manager
    stdio.initialize();

    // Initialize program/thread manager
    programManager.initialize();

    // Start first thread
    int pid = programManager.executeThread(first_thread, nullptr, "first thread", 1);
    if (pid == -1)
    {
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
