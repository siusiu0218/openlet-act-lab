"""Job-local exact tensor cache; bounded files, no image conversion or global cache."""
import hashlib
from pathlib import Path

class ValidationCache:
    def __init__(self,dataset,indices,directory,torch,cap=3*1024**3):
        self.dataset=dataset;self.allowed=set(indices);self.directory=Path(directory)
        if len(self.allowed)!=350:raise ValueError('fixed350 cache only')
        self.directory.mkdir(exist_ok=False);self.torch=torch;self.cap=cap;self.bytes=0;self.receipts={};self.hits=0;self.misses=0
    @staticmethod
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            while b:=f.read(4*1024**2):h.update(b)
        return h.hexdigest()
    def __getitem__(self,index):
        if index not in self.allowed:raise ValueError('unplanned validation input')
        path=self.directory/(str(index)+'.pt')
        if index in self.receipts:
            if self.digest(path)!=self.receipts[index]:raise ValueError('validation cache SHA mismatch')
            self.hits+=1
            return self.torch.load(path,map_location='cpu',weights_only=False)
        raw=self.dataset[index];tmp=path.with_suffix('.tmp')
        self.torch.save(raw,tmp)
        size=tmp.stat().st_size
        if size>10*1024**2 or self.bytes+size>self.cap:raise ValueError('validation cache capacity exceeded')
        tmp.rename(path);self.receipts[index]=self.digest(path);self.bytes+=size;self.misses+=1
        return raw
    def stats(self):return dict(entries=len(self.receipts),bytes=self.bytes,hits=self.hits,misses=self.misses,cap=self.cap)
