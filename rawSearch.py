import json
class TFIDF():
    def __init__(self):
        # Load data
        with open('data/ds.json', 'r') as file:
            self.ds = json.load(file)
        with open('data/docs.json', 'r') as file:
            self.docs = json.load(file)["docs"]
        with open('data/tf_idf_list.json', 'r') as file:
            self.tf_idf = json.load(file)
        print("Finish loading data")

    def search(self, q, k):
        # Search documents using TF-IDF
        results = []
        finals = []

        for i in range(len(self.docs)):
            score = 0
            for w in q.split():
                w = w.lower()
                score += self.tf_idf[w][str(i)] / self.ds[str(i)]
            finals.append((score, i))

        filter = sorted(finals, key= lambda x: -x[0])[:k]

        for item in filter:
            temp = {
                "doc_id" : item[1],
                "score" : item[0],
                "text" : self.docs[item[1]]
            }
            results.append(temp)

        return results


if __name__ == '__main__':
    tf_idf = TFIDF()
    print(tf_idf.search("Miền Bắc Việt Nam", 1))