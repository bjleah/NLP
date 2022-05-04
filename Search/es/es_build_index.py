
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from log import get_logger
logger = get_logger()


##es connect

NF_HOST = ["clusterMaster", "clusterWorker1", "clusterWorker2"]
ES_PORT = 9201

def es_connect(mode = "cluster"):
    if mode == "local":
        return Elasticsearch()
    elif mode == "cluster":
        try:
            return Elasticsearch(hosts=NF_HOST[0], port=ES_PORT, timeout=30, max_retries=3, retry_on_timeout=True)
        except Exception as e:
            logger.error(e)

##es 建表
def es_build_index(index, data = None):
    es = es_connect()
    if not isinstance(es, Elasticsearch):
        logger.error("Elasticsearch error")
        return

    es.indices.create(index=index, ignore=400)##即存在也创建

    if isinstance(data, dict):#单条数据插入
        es.index(index=index, doc_type='_doc', body=data)
    elif isinstance(data, list):#多条数据插入
        es.bulk(index=index, doc_type='_doc', body=data)
    es.transport.close()
    return

##es取数据
def search(index = 'detection_result', time = '7d', size = 10000):
    es = es_connect()
    ###DSL查询体（domain specific search）
    query_res = {"query": {
                            "bool": {
                                "minimum_should_match": 2,
                                "should": [
                                    {
                                        "range": {
                                            "time": {
                                                "gte": "2021-11-05T00:00.000Z",
                                                "lt": "2021-11-05T23:59.000Z"
                                            }
                                        }
                                    },
                                    {
                                        "match_phrase": {
                                            "model": "10025"
                                        }
                                    }
                                ]
                            }
                        }
                    }
    data = [i for i in helpers.scan(es, index= index, scroll= time, query=query_res, size=size)]
    #scroll 持分页查询，自动排序，并把查询结果返回, size 指定结果数据中共返回多少条数据
    es.transport.close()
    return data






