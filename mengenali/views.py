# Create your views here.
import logging
from os import path
import json

from django.core.files.storage import default_storage, get_storage_class
from django.http import HttpResponse, HttpResponseNotFound, HttpResponseBadRequest, HttpResponseNotAllowed, JsonResponse

from kawalc1 import settings
from django.views.static import serve as static_serve

from mengenali import registration, processprobs, io
from mengenali import extraction
from urllib import request
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from datetime import datetime

from mengenali.processprobs import print_outcome, print_outcome_parsable


def index(request):
    return static_serve(request, 'index.html', '/home/sjappelodorus/verifikatorc1/static')


def handle_uploaded_file(f, filename):
    target_location = path.join(settings.STATIC_DIR, 'upload/' + filename)
    default_storage.save(target_location, f)


def download_file(uri, target_path):
    file_to_get = request.urlopen(uri)
    with open(path.join(settings.STATIC_DIR, target_path + '/' + path.basename(uri)), 'wb') as downloaded_file:
        downloaded_file.write(file_to_get.read())


@csrf_exempt
def extract(request):
    filename = request.GET.get("filename", "")
    output = extraction.extract(filename, settings.STATIC_DIR, path.join(settings.STATIC_DIR, 'extracted'),
                                settings.STATIC_DIR,
                                load_config(request.GET.get("configFile", 'digit_config_pilpres_2019.json')))
    return JsonResponse(output)


@csrf_exempt
def extract_tps(request, kelurahan, tps, filename):
    config_file = request.GET.get('configFile', 'digit_config_pilpres_2019.json').lower()
    config = load_config(config_file)
    base_url = request.GET.get('baseUrl',
                               f'https://storage.googleapis.com/kawalc1/static/transformed/{kelurahan}/{tps}/')
    file_path = path.join(settings.STATIC_DIR, f'transformed/{kelurahan}/{tps}/{filename}')

    output = extraction.extract2(file_path, settings.STATIC_DIR,
                                 path.join(settings.STATIC_DIR, f'transformed/{kelurahan}/{tps}/extracted'),
                                 settings.STATIC_DIR,
                                 load_config(request.GET.get("configFile", 'digit_config_pilpres_2019.json')))
    return JsonResponse(output)


def load_config(config_file_name):
    with open(path.join(settings.DATASET_DIR, config_file_name)) as config_file:
        config = json.load(config_file)
    return config


@csrf_exempt
def get_probabilities_result_parsable(request):
    json_data = json.loads(request.body.decode('utf-8'))
    outcomes = processprobs.get_possible_outcomes_for_config(load_config(json_data["configFile"]),
                                                             json_data["probabilities"], settings.CATEGORIES_COUNT, print_outcome_parsable)
    results = []
    for outcome in outcomes:
        results.append(outcome)
    output = {'probabilityMatrix': outcomes}

    return JsonResponse(output)


@csrf_exempt
def get_probabilities_result(request):
    json_data = json.loads(request.body.decode('utf-8'))

    logging.info(str(json_data))

    outcomes = processprobs.get_possible_outcomes_for_config(load_config(json_data["configFile"]),
                                                             json_data["probabilities"], settings.CATEGORIES_COUNT, print_outcome)
    results = []
    for outcome in outcomes:
        results.append(outcome)
    output = {'probabilityMatrix': outcomes}

    return JsonResponse(output)


def get_reference_form(config_file_name):
    config = load_config(config_file_name)
    return path.join(settings.DATASET_DIR, config["referenceForm"])


def find_number(output, name):
    for prob_sets in output['probabilityMatrix']:
        for prob in prob_sets:
            for (k, v) in prob.items():
                if type(v) is dict and v['shortName'] == name:
                    return v['number']


def get_outcome(output):
    return {
        'prabowo': find_number(output, 'prabowo'),
        'jokowi': find_number(output, 'jokowi'),
        'jumlah': find_number(output, 'jumlah'),
        'tidakSah': find_number(output, 'tidakSah'),
        'confidence': output['probabilityMatrix'][0][0]['confidence']
    }


def align(request, kelurahan, tps, filename):
    try:
        config_file = request.GET.get('configFile', 'digit_config_pilpres_2019.json').lower()
        a = __do_alignment(filename, kelurahan, request, tps, get_reference_form(config_file))
        output = {**a}

    except Exception as e:
        logging.exception("failed 'download/<int:kelurahan>/<int:tps>/<str:filename>'")
        output = {'transformedUrl': None, 'success': False}

    return JsonResponse(output)


def __do_alignment(filename, kelurahan, request, tps, reference_form):
    store_files = json.loads(request.GET.get('storeFiles', 'false').lower())
    base_url = request.GET.get('baseUrl', f'https://storage.googleapis.com/kawalc1/firebase/{kelurahan}/{tps}/')
    if not store_files:
        io.storage = get_storage_class('inmemorystorage.InMemoryStorage')()
    else:
        io.storage = get_storage_class(
            'django.core.files.storage.FileSystemStorage')() if settings.LOCAL else get_storage_class(
            'storages.backends.gcloud.GoogleCloudStorage')()
    url = f'{base_url}/{filename}'
    output_path = path.join(settings.STATIC_DIR, 'transformed')
    from datetime import datetime
    lap = datetime.now()
    a = json.loads(
        registration.register_image(url, reference_form, output_path, None, f'{kelurahan}/{tps}'))
    logging.info("1: Register  %s", (datetime.now() - lap).total_seconds())
    return a


def download(request, kelurahan, tps, filename):
    config_file = "digit_config_pilpres_2019.json"
    loaded_config = load_config(config_file)
    try:
        maybe_extract_digits = json.loads(request.GET.get('extractDigits', 'true').lower())
        calculate_numbers = json.loads(request.GET.get('calculateNumbers', 'true').lower())
        store_files = json.loads(request.GET.get('storeFiles', 'false').lower())

        a = __do_alignment(filename, kelurahan, request, tps, get_reference_form(config_file))
        extract_digits = maybe_extract_digits or calculate_numbers
        start_lap = datetime.now()

        lap = datetime.now()
        b = extraction.extract(a['transformedUri'], settings.STATIC_DIR, path.join(settings.STATIC_DIR, 'extracted'),
                               settings.STATIC_DIR, loaded_config, store_files) if extract_digits else {"numbers": []}
        logging.info("2: Extract  %s", (datetime.now() - lap).total_seconds())
        lap = datetime.now()

        probabilities = []
        for number in b["numbers"]:
            probability_set = {"id": number["id"]}
            number_probabilities = []

            for digit_probability in number["extracted"]:
                number_probabilities.append(digit_probability["probabilities"])
            probability_set["probabilitiesForNumber"] = number_probabilities
            probabilities.insert(0, probability_set)

        c = processprobs.get_possible_outcomes_for_config(loaded_config, probabilities,
                                                          settings.CATEGORIES_COUNT, print_outcome) if calculate_numbers else {}
        logging.info("3: Probs  %s", (datetime.now() - lap).total_seconds())

        if calculate_numbers:
            del b['numbers']
            del b['signatures']
            b['probabilityMatrix'] = c

        output = {**a, **b}

        outcome = get_outcome(output)
        output['outcome'] = outcome

        if not store_files:
            io.storage.delete("static")
            del output['configFile']
            del output['probabilityMatrix']
            del output['digitArea']
            del output['transformedUri']
            del output['transformedUrl']
        output['success'] = bool(outcome['confidence'] > 0.8)

        output['duration'] = (datetime.now() - start_lap).total_seconds()
        output['configFile'] = config_file

    except Exception as e:
        logging.exception("failed 'download/<int:kelurahan>/<int:tps>/<str:filename>'")
        output = {'transformedUrl': None, 'success': False}

    return JsonResponse(output)


@csrf_exempt
def transform(request):
    if request.method == 'POST':
        filename = request.POST.get("flowFilename", "")
        handle_uploaded_file(request.FILES['file'], filename)

        try:
            config_file = request.POST.get("configFile", "")
            output = json.loads(
                registration.process_file(None, 1, settings.STATIC_DIR, filename, get_reference_form(config_file),
                                          config_file))
            output["configFile"] = config_file
        except Exception as e:
            logging.exception("this is not good!")
            output = {'transformedUrl': None, 'success': False}
        return JsonResponse(output)
    else:
        return HttpResponseNotFound('Method not supported')


def lazy_load_reference_form(form_uri):
    if path.isfile(path.join(settings.DATASET_DIR, form_uri)):
        return path.join(settings.DATASET_DIR, form_uri)
    # file_name = path.basename(form_uri)
    # uploaded_location = path.join(settings.STATIC_DIR, 'datasets/' + file_name)
    # if path.isfile(uploaded_location):
    #     return uploaded_location
    # download_file(form_uri, "datasets")
    # return uploaded_location
    raise Exception("supplied reference form " + str(form_uri) + " not found.")


def custom(request):
    if request.method == 'POST':
        req = json.loads(request.body)

        if "scanURL" not in req:
            return HttpResponseBadRequest("scanURL parameter required")
        if "config" not in req:
            return HttpResponseBadRequest("config parameter required")
        if "configName" not in req:
            return HttpResponseBadRequest("configName parameter required")

        scan_url = req["scanURL"]
        posted_config = req["config"]
        config_name = req["configName"]
        try:
            output = process_form(config_name, posted_config, scan_url)

        except Exception as err:
            logging.exception(err)
            output = json.dumps({'transformedUrl': None, 'success': False, 'error': str(err)}, separators=(',', ':'))

        return HttpResponse(output)
    else:
        return HttpResponseNotAllowed('Method not supported')


def process_form(config_name, posted_config, scan_url):
    ref_form = lazy_load_reference_form(posted_config["referenceForm"])
    with open(path.join(settings.STATIC_DIR, 'datasets/' + config_name), "w") as outfile:
        json.dump(posted_config, outfile, separators=(',', ':'), indent=4)
    before_dot = scan_url.find('.jpg') - 1
    url_as_list = list(scan_url)
    try:
        start_page = int(url_as_list[before_dot])
        pages_in_order = range(1, 6)
        pages_in_order.remove(start_page)
        pages_in_order.insert(0, start_page)
    except ValueError:
        pages_in_order = []
    transform_output = None
    for page in pages_in_order:
        url_as_list[before_dot] = str(page)
        url_to_download = ''.join(url_as_list)
        download_file(url_to_download, "upload")
        try:
            registered_file = registration.process_file(None, 1, settings.STATIC_DIR,
                                                        path.basename(url_to_download),
                                                        ref_form,
                                                        config_name)
        except Exception as err:
            logging.warning(Exception, err)
            continue
        transform_output = json.loads(registered_file)
        if transform_output["success"]:
            break
    if transform_output is None:
        download_file(scan_url, "upload")
        registered_file = registration.process_file(None, 1, settings.STATIC_DIR,
                                                    path.basename(scan_url),
                                                    ref_form,
                                                    config_name)
        transform_output = json.loads(registered_file)
    transformed_url = transform_output["transformedUrl"]
    extracted = json.loads(extraction.extract("transformed/" + transformed_url, settings.STATIC_DIR,
                                              path.join(settings.STATIC_DIR, 'extracted'),
                                              settings.STATIC_DIR, posted_config))
    digit_area = extracted["digitArea"]
    numbers = extracted["numbers"]
    probabilities = []
    for number in numbers:
        probability_set = {"id": number["id"]}
        number_probabilities = []

        for digit_probability in number["extracted"]:
            number_probabilities.append(digit_probability["probabilities"])
        probability_set["probabilitiesForNumber"] = number_probabilities
        probabilities.insert(0, probability_set)
    outcomes = processprobs.get_possible_outcomes_for_config(posted_config,
                                                             probabilities, settings.CATEGORIES_COUNT, print_outcome)
    results = []
    for outcome in outcomes:
        results.append(outcome)
    output = json.dumps({'probabilityMatrix': outcomes, 'digitArea': digit_area}, separators=(',', ':'))
    return output
