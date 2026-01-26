package com.mangatranslator.mangatranslator.services;

import com.mangatranslator.mangatranslator.dtos.JobResponseDto;
import com.mangatranslator.mangatranslator.models.Job;
import com.mangatranslator.mangatranslator.repositories.JobRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class JobService {
    @Autowired
    private JobRepository jobRepository;

    public JobResponseDto getJob(String id){
        JobResponseDto jobResponseDto = new JobResponseDto();
        Job job = this.jobRepository.findById(id).orElseThrow(() -> new RuntimeException("Not found"));
        jobResponseDto.setId(job.getId());
        jobResponseDto.setProgress(job.getProgress());
        jobResponseDto.setStatus(job.getStatus());
        jobResponseDto.setDocumentId(job.getDocumentId());
        jobResponseDto.setCreatedAt(job.getCreatedAt());
        jobResponseDto.setSourceLang(job.getSourceLang());
        jobResponseDto.setTargetLang(job.getTargetLang());
        return jobResponseDto;
    }

    public String downloadFile(String id){
        Job job = this.jobRepository.findById(id).orElseThrow(() -> new RuntimeException("Not found"));
        if(!job.getStatus().equals("DONE")){
            throw  new RuntimeException("Translation is still not done");
        }
        return "Subbed";
    }
}
