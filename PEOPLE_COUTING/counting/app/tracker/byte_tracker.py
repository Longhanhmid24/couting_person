"""Compact ByteTrack with high/low confidence association and Kalman prediction."""
from dataclasses import dataclass
from .kalman_filter import KalmanFilter
from .matching import iou_matrix, linear_assignment

@dataclass
class TrackResult:
    track_id:int; bbox:tuple; score:float; cls:object; confirmed:bool=True; matched:bool=True; lost:int=0

class _Track:
    def __init__(self,track_id,det,kf):
        box,self.score,self.cls=det; self.track_id=track_id; self.mean,self.covariance=kf.initiate(box)
        self.hits=1; self.lost=0; self.confirmed=True; self.matched=True
    @property
    def bbox(self): return KalmanFilter.to_xywh(self.mean)

class BYTETracker:
    def __init__(self,high_thresh=.4,low_thresh=.1,new_track_thresh=.5,match_thresh=.8,track_buffer=25):
        self.high_thresh=high_thresh; self.low_thresh=low_thresh; self.new_track_thresh=new_track_thresh
        self.match_thresh=match_thresh; self.track_buffer=max(1,int(track_buffer)); self.kf=KalmanFilter(); self.tracks=[]; self._next_id=1
        self.last_high_matches=0; self.last_low_matches=0

    def retire(self, track_ids):
        if not track_ids: return
        s = set(track_ids)
        self.tracks = [t for t in self.tracks if t.track_id not in s]

    def _associate(self,tracks,dets,indices):
        if not tracks or not indices: return [],list(range(len(tracks))),list(indices)
        track_boxes=[t.bbox for t in tracks]
        det_boxes=[dets[i][0] for i in indices]
        overlap=iou_matrix(track_boxes,det_boxes)
        cost=1.0-overlap
        # At low detection cadence a person can move farther than one box width
        # between detections. Use Kalman-predicted center proximity as a gated
        # secondary affinity so valid fast moves do not create a fresh ID.
        for ti,track_box in enumerate(track_boxes):
            tcx=track_box[0]+track_box[2]/2; tcy=track_box[1]+track_box[3]/2
            for ci,det_box in enumerate(det_boxes):
                dcx=det_box[0]+det_box[2]/2; dcy=det_box[1]+det_box[3]/2
                scale=max(track_box[2],track_box[3],det_box[2],det_box[3],1.0)
                distance=((tcx-dcx)**2+(tcy-dcy)**2)**.5
                ratio=max(track_box[2]*track_box[3],det_box[2]*det_box[3])/max(1.0,min(track_box[2]*track_box[3],det_box[2]*det_box[3]))
                if distance>1.5*scale or ratio>4.0:
                    cost[ti,ci]=1e6
                else:
                    proximity=max(0.0,1.0-distance/(1.5*scale))
                    cost[ti,ci]=min(cost[ti,ci],1.0-(.2+.8*proximity))
        pairs,_,_=linear_assignment(cost); accepted=[]
        for ti,ci in pairs:
            if cost[ti,ci]<=self.match_thresh: accepted.append((ti,indices[ci]))
        mt={i for i,_ in accepted}; md={j for _,j in accepted}
        return accepted,[i for i in range(len(tracks)) if i not in mt],[i for i in indices if i not in md]

    def _apply(self,track,det):
        box,track.score,track.cls=det; track.mean,track.covariance=self.kf.update(track.mean,track.covariance,box)
        track.hits+=1; track.lost=0; track.matched=True; track.confirmed=True

    def update(self,detections,dt=1.0):
        dets=[(tuple(map(float,b)),float(score),cls) for b,score,cls in detections
              if len(b)==4 and score>=self.low_thresh]
        for track in self.tracks:
            track.mean,track.covariance=self.kf.predict(track.mean,track.covariance,dt)
            track.lost+=1; track.matched=False
        high=[i for i,d in enumerate(dets) if d[1]>=self.high_thresh]
        low=[i for i,d in enumerate(dets) if self.low_thresh<=d[1]<self.high_thresh]
        high_pairs,unmatched_tracks,_=self._associate(self.tracks,dets,high)
        for ti,di in high_pairs: self._apply(self.tracks[ti],dets[di])
        self.last_high_matches=len(high_pairs)
        remaining=[self.tracks[i] for i in unmatched_tracks if self.tracks[i].confirmed]
        low_pairs,_,_=self._associate(remaining,dets,low)
        for ti,di in low_pairs: self._apply(remaining[ti],dets[di])
        self.last_low_matches=len(low_pairs)
        matched_dets={di for _,di in high_pairs}|{di for _,di in low_pairs}
        for di,det in enumerate(dets):
            if di not in matched_dets and det[1]>=self.new_track_thresh:
                self.tracks.append(_Track(self._next_id,det,self.kf)); self._next_id+=1
        expired=[t.track_id for t in self.tracks if t.lost>self.track_buffer]
        self.tracks=[t for t in self.tracks if t.lost<=self.track_buffer]
        out=[TrackResult(t.track_id,t.bbox,t.score,t.cls,t.confirmed,t.matched,t.lost)
             for t in self.tracks if t.confirmed]
        return out,expired

    def predict(self,dt=1.0):
        for t in self.tracks: t.mean,t.covariance=self.kf.predict(t.mean,t.covariance,dt); t.matched=False
        expired=[t.track_id for t in self.tracks if t.lost>self.track_buffer]
        self.tracks=[t for t in self.tracks if t.lost<=self.track_buffer]
        return [TrackResult(t.track_id,t.bbox,t.score,t.cls,t.confirmed,False,t.lost) for t in self.tracks if t.confirmed],expired
