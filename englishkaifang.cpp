/*数据结构大作业，基于哈希表的关键词上下文查找。文本从txt文件中提取，也可以输入新文章。
目前只实现了英文部分，且尚有许多不足之处*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <chrono>
using namespace std;

int p = 500000;

struct HashEntry {
    string key;
    vector<string> sentences;
    bool occupied = false;
};

vector<string> splitIntoSentences(const string& text) {
    vector<string> sentences;
    string sentence;
    for (char ch : text) {
        if ((unsigned char)ch > 127){
            continue;
        }
        sentence += ch;
        if (ch == '.' || ch == '?' || ch == '!') {
            sentences.push_back(sentence);
            sentence.clear();
        }
    }
    if (!sentence.empty()) {
        sentences.push_back(sentence);
    }
    for (string& cur : sentences) {
        if (cur[0] == ' ') {
            cur.erase(0, 1);
        }
    }
    return sentences;
}

string toLowerCase(const string& str) {
    string lowerStr = str;
    transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return lowerStr;
}

int hashvalue(const string& word){
	int ans = 0;
    for(int i = 0; i < (int)word.size(); ++i)
        ans = (ans + word[i] - 'a' + 1) % p;
    return ans;
}

int bkdrhashvalue(const string& word) {
    int seed = 31;
    int ans = 0;
    for(int i = 0; i < (int)word.size(); ++i)
        ans = ans * seed + word[i] - 'a' + 1;
    return (ans & 0x7fffffff);
}

vector<HashEntry> buildHashTable(const vector<string>& sentences) {
    vector<HashEntry> hashTable(p);
    for (const string& sentence : sentences) {
        istringstream iss(sentence);
        string word;
        set <string> notrepeatedword;
        while (iss >> word) {
            word.erase(remove_if(word.begin(), word.end(), ::ispunct), word.end());
            if(notrepeatedword.find(toLowerCase(word)) != notrepeatedword.end()){
                continue;
            }
            else if(any_of(word.begin(), word.end(), ::isdigit)){
                continue;
            }
            else{
                string lowerword = toLowerCase(word);
                notrepeatedword.insert(lowerword);
                int index = hashvalue(lowerword);
                while(hashTable[index].occupied && hashTable[index].key != lowerword){
                    index = (index + 1) % p; 
                }
                if(!hashTable[index].occupied){
                    hashTable[index].key = lowerword;
                    hashTable[index].occupied = true;
                }
                hashTable[index].sentences.push_back(sentence);
            }
        }
    }
    return hashTable;
}

int main() {
    string article;
    ostringstream buffer;
    ifstream file("bib.txt");
    if(!file.is_open()){
        cout << "cannot open the file" << endl;
    }
    string line;
    while(getline(file, line)){
        buffer << line;
    }
    file.close();
    article = buffer.str();
    //cout<<"please input article, press ctrl + d + enter to end input"<<endl;
    //getline(cin, article);
    //buffer << cin.rdbuf(); 
    //article = buffer.str();     
    //article.erase(remove(article.begin(), article.end(), '\n'), article.end());
    auto start = chrono::high_resolution_clock::now();
    vector<string> sentences = splitIntoSentences(article);
    vector<HashEntry> hashtable = buildHashTable(sentences);
    auto end = chrono::high_resolution_clock::now();
    auto buildDuration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "hash table built in " << buildDuration.count() << " ms" << endl;
    while (true) {
        cout << "options:\n1. query a word\n2. update the article\n0. exit" << endl;
        string choice;
        cin >> choice;
        cin.ignore(); 

        if (choice == "0") {
            break;
        } 

        else if (choice == "1") {
            cout << "please input query:" << endl;
            string query;
            cin >> query;

            auto queryStart = chrono::high_resolution_clock::now();
            string lowerQuery = toLowerCase(query);
            int index = hashvalue(lowerQuery);
            bool found = false;
            int querycount = 0;
            while (hashtable[index].occupied && querycount != p) {
                if (hashtable[index].key == lowerQuery) {
                    for (const string& consequence : hashtable[index].sentences) {
                        cout << consequence << endl;
                    }
                    found = true;
                    break;
                }
                index = (index + 1) % p;
                querycount++;
            }
            auto queryEnd = chrono::high_resolution_clock::now();
            auto queryDuration = chrono::duration_cast<chrono::microseconds>(queryEnd - queryStart);
            if (!found) {
                cout << "no sentences containing the word \"" << query << "\" were found in the article." << endl;
            }
            cout << "query executed in " << queryDuration.count() << " microseconds." << endl;
        } 

        else if (choice == "2") {
            cout << "please input the updated article:" << endl;
            //getline(cin, article);
            string newarticle;
            ostringstream newbuffer;
            newbuffer << cin.rdbuf();
            newarticle = newbuffer.str();
            newarticle.erase(remove(newarticle.begin(), newarticle.end(), '\n'), newarticle.end());
            sentences = splitIntoSentences(newarticle);
            hashtable = buildHashTable(sentences);
            cout << "article updated successfully." << endl;
        } 

        else {
            cout << "invalid choice. please try again." << endl;
        }
    }
    return 0;
}

