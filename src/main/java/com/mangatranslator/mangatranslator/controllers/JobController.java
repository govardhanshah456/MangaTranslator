package com.mangatranslator.mangatranslator.controllers;

import com.mangatranslator.mangatranslator.dtos.JobResponseDto;
import com.mangatranslator.mangatranslator.repositories.JobRepository;
import com.mangatranslator.mangatranslator.services.JobService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController("/api/jobs")
public class JobController {
    @Autowired
    private JobService jobService;

    @GetMapping("/{id}")
    public ResponseEntity<JobResponseDto> getJob(@PathVariable("id") String id){
        return new ResponseEntity<>(this.jobService.getJob(id), HttpStatus.OK);
    }

    @GetMapping("/{id}/download")
    public ResponseEntity<?> downloadFile(@PathVariable("id") String id){
        return new ResponseEntity<>(this.jobService.downloadFile(id), HttpStatus.OK);
    }
}
