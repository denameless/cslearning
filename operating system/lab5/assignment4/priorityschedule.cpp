 void ProgramManager::schedule()
{
     bool status = interruptManager.getInterruptStatus();
    interruptManager.disableInterrupt();

    if (readyPrograms.size() == 0)
    {
        interruptManager.setInterruptStatus(status);
        return;
    }

    if (running->status == ProgramStatus::RUNNING)
    {
        running->status = ProgramStatus::READY;
        running->ticks = running->priority * 10;
        readyPrograms.push_back(&(running->tagInGeneralList));
    }
    else if (running->status == ProgramStatus::DEAD)
    {
        releasePCB(running);
    }

    ListItem *highestPriorityItem = readyPrograms.front();
    PCB *highestPriorityThread = ListItem2PCB(highestPriorityItem, tagInGeneralList);
    int maxPriority = highestPriorityThread->priority;

    for (ListItem *item = readyPrograms.front(); item != nullptr; item = item->next) {
        PCB *thread = ListItem2PCB(item, tagInGeneralList);
        if (thread->priority > maxPriority) {
            highestPriorityItem = item;
            highestPriorityThread = thread;
            maxPriority = thread->priority;
        }
    }

    readyPrograms.erase(highestPriorityItem);
    highestPriorityThread->status = ProgramStatus::RUNNING;
    PCB *cur = running;
    running = highestPriorityThread;
    asm_switch_thread(cur, highestPriorityThread);
    interruptManager.setInterruptStatus(status);
}
