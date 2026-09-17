# Beyond-G server management

`ext_csh`와 `ext_csv`가 **동일한 `main` 코드**를 사용합니다. 서버 ID, SSH 계정, 기존 실험 큐 경로만 `config/`에서 다릅니다.

## 동작

| 상황 | 처리 |
| --- | --- |
| 처음 포털 대기열에 있음 | 할당될 때까지 확인. CPU는 새로 시작하지 않음 |
| GPU 할당 + 도커 SSH 정상 + GPU 확인 | 기존 GPU 큐 재개 |
| 포털 잔여 시간 ≤ 6.5시간 | 학습을 먼저 종료하지 않고 `/container/<설정한 sid>/start`로 리로드 |
| 사용하던 GPU가 회수됨 | CPU 큐 재개. 포털이 stopped이면 GPU 요청, queued이면 그대로 대기 |
| CPU 실행 중 GPU 재할당 | 도커/GPU 확인 → CPU 런처·학습 프로세스 SIGTERM → 모두 종료 확인 → GPU 큐 재개 |
| 포털 장애 또는 SSH 실패만 발생 | 회수로 단정하지 않음. 이미 실행 중인 작업을 유지하며 재확인 |

회수 확인은 **이전에 Running이었던 해당 서버의 상태가 포털에서 queued/stopped/expired 등으로 바뀌었을 때**입니다. 다른 서버의 상태나 GPU 사용률로 판단하지 않습니다. 회수 이후에는 대기열에 있어도 CPU 작업을 계속합니다.

리로드 실패 시에도 정상 GPU 작업을 유지합니다. 실패 요청은 기본 60초 후 재시도하며, 새 잔여 시간 또는 대기열 진입을 확인해야 리로드 성공으로 기록합니다. CPU 종료가 늦어져도 강제 SIGKILL하거나 GPU를 겹쳐 실행하지 않습니다.

## 서버 설정

| 프로필 | 포털 서버 ID | 기존 큐 |
| --- | --- | --- |
| `ext_csh` | `dgx-h200-1` | `/home/ext_csh/MPI_sweep/logs/mpi_tau40_ext/run_queue_imp_me_k14_s0123_{cpu,gpu}.sh` |
| `ext_csv` | `dgx-h200-2` | FQL JAX loco9 T-init-5, alpha LR 3e-4 / 1e-3 / 2e-3, seeds 0–3 |

설정은 `config/ext_csh.json`, `config/ext_csv.json`입니다. 다른 실험으로 바꾸면 `worker.cpu/gpu.command`, `cwd`, `queue_patterns`, `process_patterns`를 함께 변경합니다. 명령은 셸 문자열 대신 argv 배열입니다. CPU와 GPU 큐는 **같은 체크포인트·결과 디렉터리**를 사용해야 합니다.

필요 조건: Linux, Python 3.10+, OpenSSH, `flock`. 포털/감시 코드는 Python 표준 라이브러리만 사용합니다. 실험의 Python 환경·데이터셋은 기존 서버 환경을 사용합니다. 감시기는 GPU 컨테이너 바깥의 호스트에서 실행합니다.

- 기존 `logs/beyondg/portal.env`의 `BEYONDG_USER`, `BEYONDG_PASS`를 그대로 읽습니다. 예시는 `scripts/hosts/beyondg/portal.env.example`에 있습니다. 암호 파일은 Git에 넣지 않습니다.
- SSH 키는 각 계정의 `~/.ssh/gpubox_host`입니다.
- 도커 안에서도 설정된 실험/큐 경로가 접근 가능해야 합니다. `ext_csv`의 공통 큐 래퍼는 이 저장소 경로도 도커에 마운트되어 있어야 합니다.
- 프로세스는 현재 계정의 실제 스크립트 argv, CPU 옵션·환경, PID 시작 시각으로 확인합니다. SSH 검사 명령은 작업으로 세지 않습니다.
- 체크포인트 저장·복구는 기존 실험 코드와 큐 런처가 담당합니다. 이 저장소는 SIGTERM과 종료 확인을 조정하며, 학습 코드나 체크포인트 형식을 변경하지 않습니다. FQL 런처는 기존 `--retry-failed --detach` 경로를 유지합니다.

## main으로 전환

각 서버의 이 저장소 checkout에서 실행합니다. 새 checkout은 다음과 같이 만듭니다.

```bash
git clone -b main https://github.com/SChoish/server_management.git ~/server_management
cd ~/server_management
```

이미 checkout이 있으면:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
```

`ext_csh` 서버:

```bash
python3 scripts/hosts/beyondg/restart_watchers.py --profile ext_csh
```

`ext_csv` 서버:

```bash
python3 scripts/hosts/beyondg/restart_watchers.py --profile ext_csv
```

`restart_watchers.py`는 **현재 계정의 기존 Beyond-G 감시기만 종료**하고 공통 감시기를 시작합니다. 실행 중인 실험 큐는 전환 시점에 종료하지 않습니다. 이전 cron/systemd가 옛 감시기를 다시 띄우도록 설정되어 있다면 그 실행 경로도 공통 `ensure_beyondg_watchers.sh`로 변경합니다.

로그는 각 서버의 기존 `/home/<계정>/MPI_sweep/logs/beyondg/`에 남습니다.

- `watcher.log`: 상태 전환 및 오류
- `lease.json`: 마지막 상태 (`phase`, `portal_state`, `loss_confirmed`, `events`)
- `lease.watch.log`: 감시기 stdout/stderr
- `queue_cpu.log`, `queue_gpu.log`: 큐 실행 로그

설정만 확인하려면 아래 명령을 사용합니다. 포털 접속이나 작업 실행은 하지 않습니다.

```bash
python3 scripts/hosts/beyondg/watch_beyondg_lease.py --profile ext_csh --print-state-dir
```

프로세스를 변경하지 않고 포털 상태만 읽으려면 `--apply` 없이 실행합니다. 실행 중인 공통 감시기가 있으면 중복 실행 잠금 때문에 종료됩니다.

```bash
python3 scripts/hosts/beyondg/watch_beyondg_lease.py --profile ext_csh
```

사용자 설정은 `--config config/my-server.local.json`으로 지정할 수 있습니다. 프로필을 생략하면 로그인 계정 이름을 사용합니다. 예전 `watch_docker_port.py`와 `launch_*_gpu_in_docker.sh` 진입점도 공통 제어기로 연결되어 별도 큐를 중복 실행하지 않습니다.

## 검증

```bash
python3 -m unittest discover -s tests -v
```

포털/SSH 장애, 정확한 서버 선택, 초기 대기와 회수 후 대기의 구분, 6.5시간 리로드, CPU 종료 지연, 재할당을 검증합니다. 실제 임시 프로세스로 SIGTERM 후 체크포인트 저장 중 GPU 시작이 차단되는지도 검사합니다. 테스트는 실제 포털이나 GPU 서버에 접속하지 않습니다.
