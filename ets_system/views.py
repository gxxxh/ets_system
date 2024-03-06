from django.http import JsonResponse, HttpRequest

from ets_system import perf
from scheduler.scheduler import load_tasks, generate_gpus, SJF, JCT, MAKESPAN

all_ps = []


def hello(request: HttpRequest) -> JsonResponse:
    data = {'message': 'Hello, World!'}
    return JsonResponse(data)


def perf_model(request: HttpRequest) -> JsonResponse:
    if request.method != 'POST':
        return JsonResponse({'message': 'method not allowed'}, status=405)

    model = request.POST.get('model', None)
    batch_size = request.POST.get('batch_size', None)
    input_size = request.POST.get('input_size', None)
    dtype = request.POST.get('dtype', None)
    gpu = request.POST.get('gpu', 'T4CPUALL')
    if model is None or batch_size is None or input_size is None or dtype is None:
        return JsonResponse({'message': 'bad request, parameter can not be null'}, status=400)
    return JsonResponse({'message': 'success'}, status=200)


def list_all(request: HttpRequest) -> JsonResponse:
    res = perf.list_logs()
    return JsonResponse({'data': res}, status=200)




def list_detail(request: HttpRequest) -> JsonResponse:
    uuid = request.GET.get('uuid')
    if uuid is None:
        return JsonResponse({'message': 'bad request, parameter can not be null'}, status=400)
    res = perf.list_detail(uuid)
    if isinstance(res, str):
        return JsonResponse({'message': res}, status=400)
    else:
        return JsonResponse({'data': res}, status=200)


def get_schedule_info(request: HttpRequest) -> JsonResponse:
    time_type = request.GET.get('type', 'predict')
    if time_type not in ['predict', 'measure', 'random']:
        return JsonResponse({'message': 'bad request, parameter can not be null'}, status=400)
    tasks = load_tasks(time_type)
    gpus = generate_gpus()
    gpus = SJF(tasks, gpus)

    return JsonResponse({'message': 'success',
                         'data': {
                             'schedule': [gpu.toJSON() for gpu in gpus],
                             'JCT': JCT(gpus),
                             'makespan': MAKESPAN(gpus)
                         }
                         }, status=200, )
