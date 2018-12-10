function [status, par_workers] = create_parpool(num_cores)

    % Opens a parallel computing session with the number of workers 
    % specified by num_cores.
    %
    % 2018-04-26; Tomi Karjalainen, Jonatan Ropponen

    % Use at most as many workers as are available

    [~,nproc] = system('getconf _NPROCESSORS_ONLN');
    nproc = str2double(nproc);
    if(num_cores>nproc)
        num_cores = nproc;
    end
    
    par_workers = {};

    % Try to call parpool if more than one worker were requested for

    if(num_cores>1)
        try
            par_workers = parpool(num_cores);
            status = 1;
        catch ME
            if(strcmp(ME.identifier,'parallel:cluster:LicenseUnavailable'))
                msg = 'Could not create a parallel pool because all licences of the Parallel computing toolbox are already in use. Continuing without parallelization...\n';
                fprintf(msg);
                status = 0;
            elseif(strcmp(ME.identifier,'parallel:convenience:ConnectionOpen'))
                p = gcp;
                if(p.NumWorkers~=num_cores)
                    msg = sprintf('Found an interactive session with %.0f workers. Re-opening parpool with %.0f workers...\n',p.NumWorkers,num_cores);
                    fprintf(msg);
                    delete(p);
                    parpool(num_cores);
                    status = 1;
                else
                    fprintf('Parpool already open with %.0f workers.\n',num_cores);
                    status = 1;
                end
            else
                msg = 'Could not create a parallel pool for an unknown reason. Make sure you have a valid license for the Parallel computing toolbox. Continuing without parallelization...\n';
                fprintf(msg);
                status = 0;
            end
        end
    else
        status = 0;
    end

end