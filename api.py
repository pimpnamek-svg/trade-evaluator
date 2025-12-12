# API placeholder
from fastapi import APIRouter
import subprocess
router = APIRouter()
@router.post('/start-scanner')
def start_scanner():
    subprocess.Popen(['python','scanner.py','--use-websocket','true','--composite-scoring','true'])
    return {'status':'Scanner launched'}
