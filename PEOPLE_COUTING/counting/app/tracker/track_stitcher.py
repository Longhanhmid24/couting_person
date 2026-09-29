"""Short-lived spatial stitcher to preserve count state after track expiration."""
import time

class TrackStitcher:
    def __init__(self,retention_seconds=5.0): self.retention_seconds=retention_seconds; self.archived={}
    @staticmethod
    def _center(box): return box[0]+box[2]/2,box[1]+box[3]/2
    def archive(self,track_id,bbox,velocity,metadata=None,now=None):
        now=time.monotonic() if now is None else now
        self._prune(now)
        self.archived[track_id]={"bbox":tuple(bbox),"velocity":tuple(velocity),"metadata":metadata,"time":now}
    def recover(self,new_id,bbox,now=None):
        now=time.monotonic() if now is None else now; self._prune(now); cx,cy=self._center(bbox); area=max(1.0,bbox[2]*bbox[3]); choices=[]
        for old_id,r in self.archived.items():
            dt=max(0.0,now-r["time"]); old=r["bbox"]; ox,oy=self._center(old); vx,vy=r["velocity"]
            radius=max(old[2],old[3],bbox[2],bbox[3])*1.8; dist=((cx-ox-vx*dt*25)**2+(cy-oy-vy*dt*25)**2)**.5
            old_area=max(1.0,old[2]*old[3]); delta=abs(area-old_area)/max(area,old_area)
            if dist<radius and delta<.4: choices.append((dist/radius+delta,old_id))
        if not choices: return None
        _,old_id=min(choices); return self.archived.pop(old_id)["metadata"]
    def _prune(self,now): self.archived={k:v for k,v in self.archived.items() if now-v["time"]<=self.retention_seconds}
