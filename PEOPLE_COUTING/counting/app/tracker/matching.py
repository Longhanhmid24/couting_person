"""IoU costs and dependency-free Hungarian assignment."""
import numpy as np

def iou_matrix(boxes_a, boxes_b):
    if not boxes_a or not boxes_b: return np.zeros((len(boxes_a),len(boxes_b)),dtype=np.float64)
    a=np.asarray(boxes_a,dtype=np.float64); b=np.asarray(boxes_b,dtype=np.float64)
    ax2,ay2=a[:,0]+a[:,2],a[:,1]+a[:,3]; bx2,by2=b[:,0]+b[:,2],b[:,1]+b[:,3]
    iw=np.maximum(0,np.minimum(ax2[:,None],bx2)-np.maximum(a[:,0,None],b[None,:,0]))
    ih=np.maximum(0,np.minimum(ay2[:,None],by2)-np.maximum(a[:,1,None],b[None,:,1]))
    inter=iw*ih; aa=np.maximum(a[:,2]*a[:,3],0)[:,None]; ba=np.maximum(b[:,2]*b[:,3],0)[None,:]
    return inter/np.maximum(aa+ba-inter,1e-12)

def linear_assignment(cost):
    """Return globally minimum-cost (row,col) pairs for a rectangular matrix."""
    cost=np.asarray(cost,dtype=np.float64)
    if cost.ndim!=2 or not cost.size:
        return [],list(range(cost.shape[0] if cost.ndim==2 else 0)),list(range(cost.shape[1] if cost.ndim==2 else 0))
    transposed=cost.shape[0]>cost.shape[1]; a=cost.T.copy() if transposed else cost.copy(); n,m=a.shape
    u=np.zeros(n+1); v=np.zeros(m+1); p=np.zeros(m+1,dtype=int); way=np.zeros(m+1,dtype=int)
    for i in range(1,n+1):
        p[0]=i; j0=0; minv=np.full(m+1,np.inf); used=np.zeros(m+1,dtype=bool)
        while True:
            used[j0]=True; i0=p[j0]; delta=np.inf; j1=0
            for j in range(1,m+1):
                if not used[j]:
                    cur=a[i0-1,j-1]-u[i0]-v[j]
                    if cur<minv[j]: minv[j]=cur; way[j]=j0
                    if minv[j]<delta: delta=minv[j]; j1=j
            for j in range(m+1):
                if used[j]: u[p[j]]+=delta; v[j]-=delta
                else: minv[j]-=delta
            j0=j1
            if p[j0]==0: break
        while True:
            j1=way[j0]; p[j0]=p[j1]; j0=j1
            if j0==0: break
    pairs=[(p[j]-1,j-1) for j in range(1,m+1) if p[j]]
    if transposed: pairs=[(j,i) for i,j in pairs]
    pairs=sorted(pairs); rows={r for r,_ in pairs}; cols={c for _,c in pairs}
    return pairs,[i for i in range(cost.shape[0]) if i not in rows],[j for j in range(cost.shape[1]) if j not in cols]
