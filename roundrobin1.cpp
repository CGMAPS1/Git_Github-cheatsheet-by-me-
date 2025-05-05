#include <bits/stdc++.h>
using namespace std;
struct pro
{
    int pid, bt, tt, wt, rt, let;
};
void roundrobin(int n, vector<pro> &pr, int qt)
{
    int time = 0, completed = 0;
    float ttt = 0, twt = 0;
    // enqueue all the process inside queue
    for (int i = 0; i < n; i++)
    {
        pr[i].wt = 0;
        pr[i].rt = pr[i].bt;
        pr[i].let = 0;
    }
    while (completed < n)
    {
        int alldon = 1;
        for (int i = 0; i < n; i++)
        {
            if (pr[i].rt > 0)
            {
                alldon = 0;
                pr[i].wt += time - pr[i].let;
                if (pr[i].rt > qt)
                {
                    time += qt;
                    pr[i].rt -= qt;
                }
                else
                {
                    time += pr[i].rt;
                    pr[i].rt = 0;
                    completed++;
                }
                pr[i].let = time;
            }
        }
        if (alldon)
        {
            break;
        }
    }

    for (int i = 0; i < n; i++)
    {
        pr[i].tt = pr[i].bt + pr[i].wt;
        twt += pr[i].wt;
        ttt += pr[i].tt;
    }

    for (int i = 0; i < n; i++)
    {
        cout << "P" << pr[i].pid << "\t" << pr[i].bt << "\t\t" << pr[i].wt << "\t\t" << pr[i].tt << endl;
    }
    cout << "Average waiting time:" <<twt / n << endl;
    cout << "Average turn over time:" << ttt / n << endl;
}

int main()
{
    int n, quant;
    cout << "Enter the number of process:";
    cin >> n;
    vector<pro> pr(n);
    cout << "Enter the burst time for each process:";
    for (int i = 0; i < n; i++)
    {
        pr[i].pid = i + 1;
        cout << "burst time for process " << i + 1 << ":";
        cin >> pr[i].bt;
    }
    cout << "Enter the time -quant of whole process:";
    cin >> quant;
    roundrobin(n, pr, quant);
    return 0;
}