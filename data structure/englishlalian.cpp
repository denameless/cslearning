#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <list>
#include <set>
#include <chrono>
using namespace std;

int p = 10007; //哈希表固定大小

struct HashEntry {
    string key;
    vector<string> sentences;
    HashEntry* next;
};

int bkdrhashvalue(const string& word) { 
    int seed = 31;
    int ans = 0;
    for(int i = 0; i < (int)word.size(); ++i)
        ans = ans * seed + word[i] - 'a' + 1;
    return (ans & 0x7fffffff) % p;
}

vector<string> splitIntoSentences(const string& text) { //分句
    vector<string> sentences;
    string sentence;
    for (char ch : text) {
        if ((unsigned char)ch > 127){ //忽略非ascii字符
            continue;
        }
        sentence += ch;
        if (ch == '.' || ch == '?' || ch == '!') { //句子结尾
            sentences.push_back(sentence);
            sentence.clear();
        }
    }
    if (!sentence.empty()) {
        sentences.push_back(sentence);
    }
    for (string& cur : sentences) { //去除句子开头的空格
        if (cur[0] == ' ') {
            cur.erase(0, 1);
        }
    }
    return sentences;
}

string toLowerCase(const string& str) { //大写转小写
    string lowerStr = str;
    transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return lowerStr;
}

int hashvalue(const string& word) {
    int ans = 0;
    for (int i = 0; i < (int)word.size(); ++i)
        ans = (ans + word[i] - 'a' + 1) % p;
    return ans;
}


vector<HashEntry*> buildHashTable(const vector<string>& sentences) {  //建立哈希表
    vector<HashEntry*> hashTable(p, nullptr);
    for (const string& sentence : sentences) {
        istringstream iss(sentence);
        string word;
        set<string> notrepeatedword;
        while (iss >> word) {
            word.erase(remove_if(word.begin(), word.end(), ::ispunct), word.end()); //去除标点符号
            string lowerWord = toLowerCase(word);
            if(notrepeatedword.find(toLowerCase(word)) != notrepeatedword.end()){ //单词去重
                continue;
            }
            else if(any_of(word.begin(), word.end(), ::isdigit)){ //忽略带数字的字符串
                continue;
            }
            else {
                notrepeatedword.insert(lowerWord);
                int index = hashvalue(lowerWord);
                HashEntry* entry = hashTable[index];
                bool found = false;
                while (entry != nullptr) {
                    if (entry->key == lowerWord) { //句子插入到已有的键值对
                        entry->sentences.push_back(sentence);
                        found = true;
                        break;
                    }
                    entry = entry->next;
                }
                if (!found) { //头插法插入新键值对
                    HashEntry* newentry = new HashEntry{lowerWord, {sentence}, nullptr};
                    if (hashTable[index] == nullptr) {
                        hashTable[index] = newentry;
                    } 
                    else {
                        newentry->next = hashTable[index]; 
                        hashTable[index] = newentry;
                    }
                }
            }
        }
    }
    return hashTable;
}

void deleteHashTable(vector<HashEntry*>& hashTable) { //释放内存
    for (auto& head : hashTable) {
        while (head != nullptr) {
            HashEntry* temp = head;
            head = head->next;
            delete temp;
        }
    }
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
    vector<HashEntry*> hashtable = buildHashTable(sentences);
    auto end = chrono::high_resolution_clock::now();
    auto buildDuration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "hash table built in " << buildDuration.count() << " ms" << endl;
    while (true) {
        cout << "options:\n1. query a word\n2. update the article\n0. exit" << endl;
        string choice;
        cin >> choice;
        cin.ignore(); // Clear the newline character from the input buffer

        if (choice == "0") {
            break;
        } 

        else if (choice == "1") { //查询单词
            cout << "please input query:" << endl;
            string query;
            cin >> query;

            auto queryStart = chrono::high_resolution_clock::now();
            string lowerQuery = toLowerCase(query);
            int index = hashvalue(lowerQuery);
            HashEntry* entry = hashtable[index];
            bool found = false;
            while (entry != nullptr) {
                if (entry->key == lowerQuery) {
                    for (const string& consequence : entry->sentences) {
                        cout << consequence << endl;
                    }
                    found = true;
                    break;
                }
                entry = entry->next;
            }
            auto queryEnd = chrono::high_resolution_clock::now();
            auto queryDuration = chrono::duration_cast<chrono::microseconds>(queryEnd - queryStart);
            if (!found) {
                cout << "no sentences containing the word \"" << query << "\" were found in the article." << endl;
            }
            cout << "query executed in " << queryDuration.count() << " microseconds." << endl;
        } 

        else if (choice == "2") { //更换文章
            cout << "please input the updated article:" << endl;
            //getline(cin, article);
            string newarticle;
            ostringstream newbuffer;
            newbuffer << cin.rdbuf();
            newarticle = newbuffer.str();
            newarticle.erase(remove(newarticle.begin(), newarticle.end(), '\n'), newarticle.end());
            deleteHashTable(hashtable);
            sentences = splitIntoSentences(newarticle);
            hashtable = buildHashTable(sentences);
            cout << "article updated successfully." << endl;
        }

        else {
            cout << "invalid choice. please try again." << endl;
        }
    }

    deleteHashTable(hashtable);
    return 0;
}

